#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/mpi/collectives.hpp>


using namespace std;
namespace mpi = boost::mpi;


string LOG;

using vid_t = uint32_t;
using ew_t = uint32_t;

struct Triple
{
  vid_t src, dst;
  ew_t weight;

  Triple(vid_t src = 0, vid_t dst = 0, ew_t weight = 0)
      : src(src), dst(dst), weight(weight) {}

  void write(ostream& os) const
  { os.write(reinterpret_cast<const char*>(this), sizeof(Triple)); }
};

struct StrHash
{
  size_t operator()(std::string const& s) const
  {
    size_t result = 2166136261U;
    for (auto iter = s.begin(); iter != s.end(); ++iter)
      result = 127 * result + static_cast<unsigned char>(*iter);
    return result;
  }
};


int main(int argc, char* argv[])
{
  /* Step 1: Read doc_id counts. */
  /* Step 2: Read an edge (doc_id, term_str, tf). */
  /*         Step 3: Map edges by hashing on term_str -> (doc_id, tf). */
  /*         Until all edges consumed. */
  /* Step 4: Send mapped edges to (and receive from) respective term reducers. */
  /* Step 5: Calculate term_id offsets. */
  /* Step 6: Write edges (doc_id, term_id, tf) to graph file. */
  /* Step 7: Write (term_str, term_id) mappings to term_id_map file. */


  mpi::environment env(argc, argv);
  mpi::communicator world;

  int rank = world.rank();

  constexpr int NPARTS_IN = 20;
  constexpr int NPARTS_OUT = 40;

  LOG = "[" + to_string(rank) + "]  ";

  string dir = "/datasets/suwaileh/clueweb12/b/la3";


  /* Step 1: Read doc_id offsets */

  if (rank == 0)
    cout << LOG << "Reading doc_id counts" << endl;

  vid_t doc_id_counts[NPARTS_IN] = {0};
  vid_t doc_id_offsets[NPARTS_IN] = {0};
  ifstream fin_doc_id_counts(dir + "/doc-id-map/counts");
  assert(fin_doc_id_counts.good());
  for (int i = 0; i < NPARTS_IN; i++)
    fin_doc_id_counts >> doc_id_counts[i];
  fin_doc_id_counts.close();
  for (int i = 1; i < NPARTS_IN; i++)
    doc_id_offsets[i] = doc_id_counts[i-1] + doc_id_offsets[i-1];


  /* Step 2: Read an edge (doc_id, term_str, tf). */
  /*         Step 3: Map each edge by hashing on term_str -> (doc_id, tf). */
  /*         Until all edges consumed. */

  if (rank == 0)
    cout << LOG << "Reading (doc_id, term_str, tf) edges from input graph file" << endl;

  // Input file: each file is read by two ranks; each of these two ranks reads half the file.

  ifstream fin_graph(dir + "/graph/" + (rank/2 < 10 ? "0" : "") + to_string(rank/2));
  if (not fin_graph.good())
  {
    cout << LOG << "Could not read input graph file " << rank / 2 << endl;
    exit(-1);
  }

  // Find mid-point.
  fin_graph.seekg(0, fin_graph.end);
  size_t mid_offset = fin_graph.tellg() / 2;
  fin_graph.seekg(mid_offset);
  char dummy[1024];
  fin_graph.getline(dummy, 1024);
  mid_offset = fin_graph.tellg();

  fin_graph.seekg(rank % 2 == 0 ? 0 : mid_offset);  // Rewind

  // Step 2a: First pass: map terms and count their occurences.

  if (rank == 0) cout << LOG << "Starting first pass" << endl;

  unordered_map<string, vid_t>* terms_and_counts[NPARTS_OUT];

  for (int i = 0; i < NPARTS_OUT; i++)  // Initalize
    terms_and_counts[i] = new unordered_map<string, vid_t>();

  while (not fin_graph.eof())
  {
    string term_str;
    pair<vid_t, ew_t> doc_id_tf;

    fin_graph >> doc_id_tf.first;
    fin_graph >> term_str;
    if (fin_graph.peek() == '\n') continue;  // Handle bug (empty term string)
    fin_graph >> doc_id_tf.second;

    if (not fin_graph.good() or fin_graph.eof())
      break;

    StrHash str_hash;
    size_t h = str_hash(term_str);

    (*terms_and_counts[h % NPARTS_OUT])[term_str]++;

    if (rank % 2 == 0 and fin_graph.tellg() >= mid_offset)
      break;
  }

  if (rank == 0) cout << LOG << "Completed first pass" << endl;

  // Step 2b: Second pass: map occurences.

  // Prepare optimized data structures.

  unordered_map<size_t, vector<pair<vid_t, ew_t>>>* term_occurences[NPARTS_OUT];

  for (int i = 0; i < NPARTS_OUT; i++)  // Initialize
    term_occurences[i] = new unordered_map<size_t, vector<pair<vid_t, ew_t>>>();

  vector<string>* flat_terms[NPARTS_OUT];
  for (int i = 0; i < NPARTS_OUT; i++)
    flat_terms[i] = new vector<string>();

  for (int i = 0; i < NPARTS_OUT; i++)
  {
    flat_terms[i]->reserve(terms_and_counts[i]->size());
    for (auto term : *terms_and_counts[i])
    {
      flat_terms[i]->push_back(term.first);
      StrHash str_hash;
      size_t h = str_hash(term.first);
      (*term_occurences[i])[h].reserve(term.second);
    }
  }

  if (rank == 0) cout << LOG << "Starting second pass" << endl;

  fin_graph.seekg(rank % 2 == 0 ? 0 : mid_offset);  // Rewind

  while (not fin_graph.eof())
  {
    string term_str;
    pair<vid_t, ew_t> doc_id_tf;

    fin_graph >> doc_id_tf.first;
    fin_graph >> term_str;
    if (fin_graph.peek() == '\n') continue;  // Handle bug (empty term string)
    fin_graph >> doc_id_tf.second;

    if (not fin_graph.good() or fin_graph.eof())
      break;

    doc_id_tf.first += doc_id_offsets[rank/2];

    StrHash str_hash;
    size_t h = str_hash(term_str);

    (*term_occurences[h % NPARTS_OUT])[h].push_back(doc_id_tf);

    if (rank % 2 == 0 and fin_graph.tellg() >= mid_offset)
      break;
  }

  if (rank == 0) cout << LOG << "Completed second pass" << endl;


  /* Step 4: Send mapped edges to (and receive from) respective term reducers. */

  if (rank == 0)
    cout << LOG << "Mapping edges to terms and reducing terms" << endl;

  // Convert each term-occurences map {term_hash -> [doc_id_tf, ...]} (except own)
  // into term-occurences vector [pair<term_hash, [doc_id_tf, ...]>, ...].

  vector<pair<size_t, vector<pair<vid_t, ew_t>>>>* flat_term_occurences[NPARTS_OUT];

  for (int i = 0; i < NPARTS_OUT; i++)  // Initialize
    flat_term_occurences[i] = new vector<pair<size_t, vector<pair<vid_t, ew_t>>>>();

  for (int i = 0; i < NPARTS_OUT; i++)
  {
    if (i != rank)
    {
      flat_term_occurences[i]->reserve(term_occurences[i]->size());
      for (auto term : *term_occurences[i])
        flat_term_occurences[i]->push_back(term);
      term_occurences[i]->clear();
      delete term_occurences[i];
    }
  }

  // Scatter terms from one rank (Pi) at a time.
  for (int i = 0; i < NPARTS_OUT; i++)
  {
    if (i == rank)  // sender (Pi)
    {
      // Send terms and their respective (doc_id, tf) pairs to Pj.

      for (int j = 0; j < NPARTS_OUT; j++)
      {
        if (j != rank)  // don't send to self
        {
          world.send(j, i * 100 + j, *flat_terms[j]);
          flat_terms[j]->clear();
          delete flat_terms[j];

          world.send(j, i * 200 + j, *flat_term_occurences[j]);
          flat_term_occurences[j]->clear();
          delete flat_term_occurences[j];
        }
      }
    }

    else
    {
      // Recv terms and their respective (doc_id, tf) pairs from Pi and insert into own maps.

      auto flat_terms_i = new vector<string>();
      world.recv(i, i * 100 + rank, *flat_terms_i);
      for (auto term : *flat_terms_i)
        (*terms_and_counts[rank])[term]++;
      flat_terms_i->clear();
      delete flat_terms_i;

      auto flat_term_occurences_i = new vector<pair<size_t, vector<pair<vid_t, ew_t>>>>();
      world.recv(i, i * 200 + rank, *flat_term_occurences_i);
      for (auto term : *flat_term_occurences_i)
      {
        (*term_occurences[rank])[term.first].reserve(term.second.size()
            + (*term_occurences[rank])[term.first].size());
        for (auto doc_id_tf : term.second)
          (*term_occurences[rank])[term.first].push_back(doc_id_tf);
      }
      flat_term_occurences_i->clear();
      delete flat_term_occurences_i;
    }
  }


  /* Step 5: Calculate term_id offsets. */

  if (rank == 0)
    cout << LOG << "Calculating term_id offsets" << endl;

  vid_t term_id_counts[NPARTS_OUT] = {0};
  vid_t term_id_offsets[NPARTS_OUT] = {0};
  mpi::all_gather(world, (vid_t) (*terms_and_counts[rank]).size(), term_id_counts);
  for (int i = 1; i < NPARTS_OUT; i++)
    term_id_offsets[i] = term_id_offsets[i-1] + term_id_counts[i-1];


  /* Step 6: Write edges (doc_id, term_id, tf) to graph file. */
  /* Step 7: Write (term_str, term_id) mappings to term_id_map file. */

  if (rank == 0)
    cout << LOG << "Writing (doc_id, term_id, tf) edges and "
                << "(term_str, term_id) mappings" << endl;

  // Output files
  ofstream fout_graph(dir + "/bin/clueweb12_catb.w.bin" + to_string(rank), ios::binary);
  if (not fout_graph.good())
  {
    cout << LOG << "Error creating graph output file" << endl;
    exit(-1);
  }

  ofstream fout_term_id_map(dir + "/term-id-map/" + (rank < 10 ? "0" : "") + to_string(rank));
  if (not fout_term_id_map.good())
  {
    cout << LOG << "Error creating term_id mapping file" << endl;
    exit(-1);
  }

  vid_t term_id = 1 + term_id_offsets[rank];  // One-indexed
  for (auto term : *terms_and_counts[rank])
  {
    fout_term_id_map << term.first << " " << term_id << endl;
    StrHash str_hash;
    size_t h = str_hash(term.first);
    for (auto doc_id_tf : (*term_occurences[rank])[h])
    {
      Triple e(doc_id_tf.first, term_id, doc_id_tf.second);
      e.write(fout_graph);
    }
    term_id++;
  }

  fout_graph.close();
  fout_term_id_map.close();

  terms_and_counts[rank]->clear();
  delete terms_and_counts[rank];

  term_occurences[rank]->clear();
  delete term_occurences[rank];

  world.barrier();


  if (rank == 0)
    cout << LOG << "Done" << endl;

  return 0;
}
