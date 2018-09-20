#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>


using namespace std;


int main(int argc, char* argv[])
{
  cout << "Convert graph from binary (with optional header: uint:n uint:m ulong:nnz) "
       << "pairs (uint:i uint:j) or triples (uint:i uint:j uint:w) "
       << "to Matrix Market (with optional header: n m nnz) "
       << "pairs (i j) or triples (i j w). "
       << endl;

  cout << "Usage: " << argv[0] << " <filepath_in> <filepath_out> "
       << endl << "\t [-hi[o]]  read in header [and write [o]ut]"
       << endl << "\t [-wi]     read in edge weights (must be int)"
       << endl << "\t [-wo[r]]  write out edge weights (int) "
       << endl << "\t           (by default 1, or rand [1,128] if [r])."
       << endl;

  if (argc < 3)
    return 1;

  string fpath_in = argv[1];
  string fpath_out = argv[2];

  bool header_in    = false;
  bool header_out   = false;
  bool weights_in   = false;
  bool weights_out  = false;
  bool weights_int  = true;
  bool weights_rand = false;

  for (auto i = 3; i < argc; i++)
  {
    if (string(argv[i]) == "-hi" or string(argv[i]) == "-hio")
      header_in = true;
    if (string(argv[i]) == "-hio")
      header_out = true;
    if (string(argv[i]) == "-wi")
      weights_in = true;
    if (string(argv[i]) == "-wo" or string(argv[i]) == "-wor")
      weights_out = true;
    if (string(argv[i]) == "-wor")
      weights_rand = true;
  }

  //weights_rand = not weights_in and weights_out;

  ifstream fin(fpath_in.c_str(), ios::binary);
  ofstream fout(fpath_out.c_str());

  // Read/write header
  uint32_t n, m;
  uint64_t nnz = 0;
  if (header_in)
  {
    fin.read(reinterpret_cast<char*>(&n),   sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&m),   sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&nnz), sizeof(uint64_t));
    cout << "Header: " << n << " " << m << " " << nnz << endl;

    // Write header
    if (header_out)
      fout << n << " " << m << " " << nnz << endl;
  }

  // Read/write pairs/triples
  uint32_t i, j;
  uint32_t wi = 1;
  srand(0);
  while (!fin.eof())
  {
    fin.read(reinterpret_cast<char*>(&i), sizeof(uint32_t));
    if (fin.eof()) break;
    fin.read(reinterpret_cast<char*>(&j), sizeof(uint32_t));

    fout << i << " " << j;
    if (weights_in)
      fin.read(reinterpret_cast<char*>(&wi), sizeof(uint32_t));
    if (weights_out)
    {
      if (weights_rand)
        wi = (uint32_t) 1 + (rand() % 128);
      fout << " " << wi;
    }
    fout << endl;
  }

  fin.close();
  fout.close();
}
