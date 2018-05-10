#ifndef GRAPHSIM_QUERY_H
#define GRAPHSIM_QUERY_H

#include <cassert>
#include <fstream>
#include <vector>
#include <iostream>


using namespace std;


void read_labels(string filepath, vector<string>& labels)
{
  ifstream fin(filepath);
  while (not fin.eof())
  {
    string label;
    fin >> label;
    labels.push_back(label);
  }
  fin.close();
}


struct Query
{
  int size = 0;                  // num of vertices
  vector<string> labels;         // vertex labels
  vector<vector<int>> children;  // adjacency list

  Query(string qgl_filepath, string qgm_filepath)
  {
    read_labels(qgl_filepath, labels);

    size = labels.size();
    children.resize(size);

    ifstream fin_mtx(qgm_filepath);
    while (not fin_mtx.eof())
    {
      int i, j;
      fin_mtx >> i;
      if (fin_mtx.eof()) break;
      fin_mtx >> j;
      children[i].push_back(j);
    }
    fin_mtx.close();
  }
};


#endif
