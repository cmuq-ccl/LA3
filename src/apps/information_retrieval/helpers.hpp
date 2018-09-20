#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <fstream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include "utils/log.h"

using namespace std;


size_t count_lines(string& filepath)
{
  size_t num_lines = 0;
  ifstream fin(filepath);
  if (not fin.good())
    LOG.fatal("Could not read file %s \n", filepath.c_str());
  while (not fin.eof())
  {
    constexpr size_t BUFF_SIZE = 2048;
    char buffer[BUFF_SIZE];
    fin.getline(buffer, BUFF_SIZE);
    if (strlen(buffer) > 0) num_lines++;
    if (fin.eof())
      break;
  }
  fin.close();
  return num_lines;
}


void load_labels(string& filepath, string& label_data, vector<size_t>& label_ptrs)
{
  size_t num_lines = count_lines(filepath);
  label_ptrs.resize(num_lines);
  ifstream fin(filepath);
  string label;
  for (auto i = 0; i < num_lines; i++)
  {
    label_ptrs[i] = label_data.size();
    label.clear();
    fin >> label;
    uint32_t term_id = 0;
    fin >> term_id;
    label_data += label + " ";
    label_data[label_data.size() - 1] = '\0';
  }
  fin.close();
  label_data.shrink_to_fit();
}


void load_queries(string& filepath, vector<vector<string>>& queries)
{
  size_t num_lines = count_lines(filepath);
  queries.resize(num_lines);
  ifstream fin(filepath);
  for (auto i = 0; i < num_lines; i++)
  {
    constexpr size_t BUFF_SIZE = 2048;
    char buffer[BUFF_SIZE];
    fin.getline(buffer, BUFF_SIZE);
    if (fin.eof())
      break;
    char* token = strtok(buffer, " ");
    while (token)
    {
      queries[i].push_back(string(token));
      token = strtok(nullptr, " ");
    }
  }
  fin.close();
}


void prepare_queries(string& filepath_term_labels,
                     string& filepath_queries, vector<vector<uint32_t>>& queries,
                     uint32_t terms_offset = 0)
{
  if (Env::is_master)
  {
    LOG.info("Loading queries ... \n");
    vector<vector<string>> orig_queries;
    load_queries(filepath_queries, orig_queries);
    LOG.info("Total queries: %lu \n", orig_queries.size());

    LOG.info("Loading term labels ... \n");
    string label_data;
    vector<size_t> label_ptrs;
    load_labels(filepath_term_labels, label_data, label_ptrs);
    LOG.info<false>("Total labels: %lu (%lu bytes) \n", label_ptrs.size(), label_data.size());

    LOG.info("Creating term ID mappings ... \n");
    unordered_map<string, uint32_t> term_id_map;
    for (auto i = 0; i < label_ptrs.size(); i++)
    {
      string term(&label_data.data()[label_ptrs[i]]);
      term_id_map[term] = i + 1;  // zero-indexed (labels) to one-indexed (vertices)
      term_id_map[term] += terms_offset;
    }
    label_data.clear();
    label_data.shrink_to_fit();
    label_ptrs.clear();
    label_ptrs.shrink_to_fit();

    LOG.info("Mapping query terms to term IDs ... \n");
    queries.resize(orig_queries.size());
    for (auto i = 0; i < queries.size(); i++)
    {
      queries[i].resize(orig_queries[i].size());
      for (auto j = 0; j < queries[i].size(); j++)
        queries[i][j] = term_id_map[orig_queries[i][j]];
    }
    term_id_map.clear();
    unordered_map<string, uint32_t>().swap(term_id_map);
    orig_queries.clear();
    orig_queries.shrink_to_fit();
  }

  LOG.info("Broadcasting mapped queries ... \n");
  boost::mpi::communicator world;
  boost::mpi::broadcast(world, queries, 0);
}
