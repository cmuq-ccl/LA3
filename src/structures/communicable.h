#ifndef COMMUNICABLE_H
#define COMMUNICABLE_H


/**
 * Offers MPI-based communication interface: isend/irecv() with isend/irecv_postprocess().
 * Expects the derived class to have functions prefixed by Array:: below.
 *
 * Why mixin-style and not a base class for Array?
 *       Because otherwise we will need virtual functions to call derived class functions.
 *       Of course, an alternative would be templated-inheritence from Communicable<Serialization>
 *       where the Serialization class provides the needed (protected) interfaces..
 *       But we also use count() and size()..
 *
 * What issues exist with this mixin-style structure?
 *       Well, we cannot (easily) have a custom and transparent constructor here.
 *       But.. so far we don't need to.
 *
 * NOTE: My use of isend/irecv() [at least in MsgOutputSegment] requires it to be possible to
 *       isend() then directly mess the up sending vector.
 *       Can the allocation here become a bottlenecks? Consider having a buffer of large blobs.
 *
 * NOTE: Should we allow or disallow calling isend_postprocess() from a different object to the
 *       one that created the blob?
 **/


template <class Array>
class Communicable : public Array
{
public:
  /* Inherit all constructors of base. */
  using Array::Array;

  void* new_blob()
  {
    return Array::new_blob(blob_nbytes_max());
  }

  void delete_blob(void* blob)
  {
    Array::delete_blob(blob);
  }

  uint32_t blob_nbytes(uint32_t count)
  {
    return Array::blob_nbytes(count);
  }

  uint32_t blob_nbytes_tight()
  {
    return Array::blob_nbytes(Array::count());
  }

  uint32_t blob_nbytes_max()
  {
    return Array::blob_nbytes(Array::size());
  }

  template <bool destructive = false>
  void* isend(int32_t rank, int32_t tag, MPI_Comm comm, MPI_Request* request)
  {
    void* blob = nullptr;
    uint32_t nbytes;

    // "IF" this is determined statically, the compiler should optimize this branch away.
    if (std::is_base_of<Serializable, typename Array::Type>::value)
    {
      // We will create the blob dynamically during serialization (passing ptr by reference).
      nbytes = Array::template serialize_into<destructive>(blob);
      MPI_Isend(blob, nbytes, MPI_BYTE, rank, tag, comm, request);
    }
    else
    {
      blob = Array::new_blob(blob_nbytes_tight());
      nbytes = Array::template serialize_into<destructive>(blob);
      MPI_Isend(blob, nbytes, MPI_BYTE, rank, tag, comm, request);
    }

    if(rank != Env::rank) Env::nbytes_sent += nbytes;
    return blob;
  }

  void isend_postprocess(void* blob)
  {
    delete_blob(blob);
  }

  void* irecv(int32_t rank, int32_t tag, MPI_Comm comm, MPI_Request* request)
  {
    void* blob = nullptr;

    // "IF" this is determined statically, the compiler should optimize this branch away.
    if (std::is_base_of<Serializable, typename Array::Type>::value)
    {
      // We call MPI_Iprobe instead of MPI_Irecv, and save the MPI_Status returned by it.
      MPI_Status* status = new MPI_Status;
      status->MPI_SOURCE = rank;
      status->MPI_TAG = tag;
      int flag = 0;
      MPI_Iprobe(rank, tag, comm, &flag, status);
      *request = MPI_REQUEST_NULL;
      return status;
    }
    else
    {
      blob = new_blob();
      MPI_Irecv(blob, blob_nbytes_max(), MPI_BYTE, rank, tag, comm, request);
      return blob;
    }
  }

  void irecv_postprocess(void* blob)
  {
    Array::deserialize_from(blob);
    delete_blob(blob);
  }

  void irecv_postprocess(void* blob, uint32_t sub_size)
  {
    Array::deserialize_from(blob, sub_size);
    delete_blob(blob);
  }

  static void
  irecv_dynamic_all(std::vector<void*>& the_blobs, std::vector<MPI_Request>& the_requests)
  {
    for (auto i = 0; i < the_blobs.size(); i++)
    {
      MPI_Status* status = (MPI_Status*) the_blobs[i];
      int source = status->MPI_SOURCE;
      int tag = status->MPI_TAG;
      delete status;

      MPI_Status status_;
      MPI_Request* request = &the_requests[i];
      int count;

      MPI_Probe(source, tag, Env::MPI_WORLD, &status_);
      MPI_Get_count(&status_, MPI_BYTE, &count);

      the_blobs[i] = new char[count];

      MPI_Irecv(the_blobs[i], count, MPI_BYTE, source, tag, Env::MPI_WORLD, request);
    }
  }

  static void
  irecv_dynamic_some(std::vector<void*>& the_blobs, std::vector<MPI_Request>& the_requests)
  {
    int num_ready = 0;

    while (num_ready == 0)
    {
      for (auto i = 0; i < the_blobs.size(); i++)
      {
        void*& blob = the_blobs[i];
        MPI_Request& request = the_requests[i];

        if (blob == nullptr)
        {
          // Skip; blob already irecv_postprocessed and deleted.
          continue;
        }

        if (request != MPI_REQUEST_NULL)
        {
          // Skip; request already linked to a pending MPI_Irecv.
          // MPI_Waitsome must try checking for its completion again.
          // Also, this means blob now holds the actual blob buffer.
          num_ready++;
          continue;
        }

        // Blob holds the MPI_Status from the original MPI_Iprobe.
        MPI_Status*& status = (MPI_Status*&) the_blobs[i];
        int source = status->MPI_SOURCE;
        int tag = status->MPI_TAG;

        MPI_Status status_;
        int flag = 0;

        MPI_Iprobe(source, tag, Env::MPI_WORLD, &flag, &status_);

        if (flag)
        {
          int count;
          MPI_Get_count(&status_, MPI_BYTE, &count);

          delete status;
          blob = new char[count];

          MPI_Irecv(blob, count, MPI_BYTE, source, tag, Env::MPI_WORLD, &request);
          num_ready++;
        }
      }
    }
  }

  static void irecv_dynamic_one(void*& blob, MPI_Request*& request)
  {
    MPI_Status* status = (MPI_Status*) blob;
    int source = status->MPI_SOURCE;
    int tag = status->MPI_TAG;
    delete status;

    MPI_Status status_;
    int count = 0;

    MPI_Probe(source, tag, Env::MPI_WORLD, &status_);
    MPI_Get_count(&status_, MPI_BYTE, &count);

    blob = new char[count];

    MPI_Irecv(blob, count, MPI_BYTE, source, tag, Env::MPI_WORLD, request);
  }

};


#endif


/*
 * Performance Notes:
 *
 * TODO: Support a bcast() operation that creates the blob once, rather than many times.
 *       It should be given a vector of integer ranks _and_ a vector to push_back(req)'s.
 *       As far as I am aware, this behavior is currently only needed in ProcessedMatrix2D.
 */