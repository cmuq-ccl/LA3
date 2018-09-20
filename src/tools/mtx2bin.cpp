#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>


using namespace std;


int main(int argc, char* argv[])
{
  cout << "Convert graph from Matrix Market (with optional header: n m nnz) "
       << "pairs (i j) or triples (i j w) "
       << "to binary (with optional header: uint:n uint:m ulong:nnz) "
       << "pairs (uint:i uint:j) or triples (uint:i uint:j uint:w). "
       << endl;

  cout << "Usage: " << argv[0] << " <filepath_in> <filepath_out> "
       << endl << "\t [-hi[o]]    read in first non-comment line as header [and write [o]ut]"
       << endl << "\t [-wi]       read in edge weights (must be int or double) "
       << endl << "\t [-wo{i|d}]  write out edge weights ({i}nt/{d}ouble) "
       << endl << "\t             (rand [1,128] if none given)"
       << endl;

  if (argc < 3)
    return 1;

  string fpath_in = argv[1];
  string fpath_out = argv[2];

  bool header_in    = false;
  bool header_out   = false;
  bool weights_in   = false;
  bool weights_out  = false;
  bool weights_int  = false;
  bool weights_rand = false;

  for (auto i = 3; i < argc; i++)
  {
    if (string(argv[i]) == "-hi" or string(argv[i]) == "-hio")
      header_in = true;
    if (string(argv[i]) == "-hio")
      header_out = true;
    if (string(argv[i]) == "-wi")
      weights_in = true;
    if (string(argv[i]) == "-woi" or string(argv[i]) == "-wod")
      weights_out = true;
    if (string(argv[i]) == "-woi")
      weights_int = true;
  }

  weights_rand = not weights_in and weights_out;

  ifstream fin(fpath_in.c_str());
  ofstream fout(fpath_out.c_str(), ios::binary);

  // Skip comments
  vector<char> buffer(1024);
  char c = (char) fin.peek();
  while (c == '%' || c == '#')
  {
    fin.getline(buffer.data(), buffer.size());
    c = (char) fin.peek();
  }

  // Read/write header
  uint32_t n, m;
  uint64_t nnz = 0;
  if (header_in)
  {
    fin.seekg(-strlen(buffer.data()), ios::cur);
    //fin >> nnz >> n >> m;
    fin >> n >> m >> nnz;
    cout << "Header: " << n << " " << m << " " << nnz << endl;

    // Write header
    if (header_out)
    {
      fout.write(reinterpret_cast<const char*>(&n),   sizeof(uint32_t));
      fout.write(reinterpret_cast<const char*>(&m),   sizeof(uint32_t));
      fout.write(reinterpret_cast<const char*>(&nnz), sizeof(uint64_t));
    }
  }

  // Read/write pairs/triples
  uint32_t i, j;
  double wd = 1;
  srand(0);
  while (!fin.eof())
  {
    fin >> i;
    if (fin.eof()) break;
    fin >> j;

    fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
    if (weights_in)
      fin >> wd;
    if (weights_out)
    {
      if (weights_rand)
        wd = 1 + (rand() % 128);
      if (weights_int)
      {
        uint32_t wi = (uint32_t) wd;
        fout.write(reinterpret_cast<const char*>(&wi), sizeof(uint32_t));
      }
      else
        fout.write(reinterpret_cast<const char*>(&wd), sizeof(double));
    }
  }

  fin.close();
  fout.close();
}
