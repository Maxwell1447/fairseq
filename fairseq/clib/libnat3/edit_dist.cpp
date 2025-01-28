#include <torch/torch.h> // @manual=//caffe2:torch_extension
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string.h>
#include <utility>
#include <algorithm>
#include <numeric>
#include <memory>
#include <unordered_map>
#include <thread>
#include <limits.h>
#include "edit_dist.h"

using namespace std;
namespace py = pybind11;

Edge::Edge(const long x, const long y, const float cost) : x(x), y(y), cost(cost)
{
}

bool Edge::operator<(Edge const &e) const
{
  return (x + y < e.x + e.y) || 
    ((x + y == e.x + e.y) && (abs(y - x) < abs(e.y - e.x)));
}
bool Edge::operator==(Edge const &e) const
{
  return (x == y) & (e.x == e.y);
}
void Edge::printEdge() const
{
  cout << "(" << x << ", " << y << "), ";
}
bool compareEdges(const Edge *e1, const Edge *e2)
{
  return *e1 < *e2;
}

Node::Node(const long index, const float cost) : index(index), cost(cost)
{
  next = {};
  prec = {};
}

void Node::addNext(Node *node)
{
  next.push_back(node);
}

bool Node::hasNext()
{
  return next.size() > 0;
}

void Node::addPrec(Node *node)
{
  prec.push_back(node);
}

bool Node::hasPrec()
{
  return prec.size() > 0;
}

vector<Edge> buildGraph(
    const long *left, const long *right,
    const long size_left, const long size_right,
    const float *idf_right,
    const long pad)
{
  vector<Edge> graph;
  unordered_map<long, list<long>> leftVertices;
  long x;
  for (long i = 0; (i < size_left) && (left[i] != pad); ++i)
  {
    x = left[i];
    auto search = leftVertices.find(x);
    if (search == leftVertices.end())
    {
      list<long> new_list;
      leftVertices.emplace(x, new_list);
    }
    leftVertices.at(x).push_back(i);
  }
  long y;
  for (long j = 0; (j < size_right) && (right[j] != pad); ++j)
  {
    y = right[j];
    auto left_same = leftVertices[y];
    for (long const &i : left_same)
    {
      Edge edge = Edge(i, j, idf_right[j]);
      graph.push_back(edge);
    }
  }
  return graph;
}

void printGraph(vector<Edge *> graph)
{
  for (Edge const *edge : graph)
  {
    edge->printEdge();
  }
  cout << "\n";
}

vector<Node> buildDAGFromGraph(vector<Edge *> &graph, const long &max_valency)
{
  vector<Node> dag;

  // initialize dag
  for (long i = 0; i < (long)graph.size() + 2; ++i)
  {
    float cost = (i == 0 || i == (long)graph.size() + 1) ? 1.f : graph[i - 1]->cost;
    dag.push_back(Node(i, cost));
  }

  // build DAG core
  long current_valency;
  for (long i = 0; i < (long)graph.size(); ++i)
  {
    current_valency = 0;
    for (long j = i + 1; (j < (long)graph.size()) && (current_valency < max_valency); ++j)
    {
      // cout << i << "," << j << ": " << current_valency << "  | ";
      if (
          (((int)graph.at(i)->x - (int)graph.at(j)->x) * ((int)graph.at(i)->y - (int)graph.at(j)->y) > 0))
      {
        current_valency++;
        dag.at(i + 1).addNext(&dag.at(j + 1));
        dag.at(j + 1).addPrec(&dag.at(i + 1));
      }
    }
    // if (i > 160)
    //   cout << graph[i]->x << "&" <<  graph[i]->y << endl;
  }
  // cout << "eos idxs: " << graph[graph.size() - 1]->x << ", " << graph[graph.size() - 1]->y << endl;
  // finalize with target/source
  // ASSUMES first and last necessarily inside
  if (dag.size() >= 4)
  {
      dag.at(graph.size()).addNext(&dag.at(graph.size() + 1));
      dag.at(graph.size() + 1).addPrec(&dag.at(graph.size()));
      dag.at(0).addNext(&dag.at(1));
      dag.at(1).addPrec(&dag.at(0));
  }
  for (long i = (long)graph.size() - 2; i >= 1; --i)
  {
    if (!dag.at(i + 1).hasNext() && dag.at(i + 1).hasPrec())
    {
      dag.at(i + 1).addNext(&dag.at(graph.size()));
      dag.at(graph.size()).addPrec(&dag.at(i + 1));
      // cout << i << "->" << "t" << endl;
    }
    if (!dag.at(i + 1).hasPrec() && dag.at(i + 1).hasNext())
    {
      dag.at(i + 1).addPrec(&dag.at(1));
      dag.at(1).addNext(&dag.at(i + 1));
      // cout << "s" << "->" << i << endl;
    }
    if (!dag.at(i + 1).hasPrec() && !dag.at(i + 1).hasNext())
    {
      dag.at(i + 1).addPrec(&dag.at(1));
      dag.at(1).addNext(&dag.at(i + 1));
      dag.at(i + 1).addNext(&dag.at(graph.size()));
      dag.at(graph.size()).addPrec(&dag.at(i + 1));
      // cout << "s" << "->" << i  << "->" << "t" << endl;
    }
  }

  return dag;
}

void printDAG(vector<Node> &dag)
{
  for (const Node &node : dag)
  {
    cout << node.index << ": ";
    for (const Node *next : node.next)
      cout << next->index << " ";
    cout << endl;
  }
}

list<long> get_single_longest_path(vector<Node> &dag)
{
  list<long> out;
  vector<long> traceback(dag.size(), -1);
  // FORWARD
  for (Node &node : dag)
  {
    node.current_length = -1.f;
    if (node.prec.size() > 0)
    {
      for (list<Node *>::iterator prec = node.prec.begin(); prec != node.prec.end(); ++prec)
        if (node.current_length < (*prec)->current_length)
        {
          node.current_length = (*prec)->current_length;
          traceback[node.index] = (*prec)->index;
        }
      node.current_length += node.cost; // +1/+IDF
    }
    else
      node.current_length = 0.f;
  }
  long index = dag.size() - 1;
  // BACKWARD
  while (index != -1)
  {
    out.emplace_front(index);
    index = traceback[index];
  }
  return out;
}

vector<list<Edge>> get_k_best(
    const long *left,
    const long *right,
    const long ls,
    const long rs,
    const float *idf_right,
    const long pad,
    const long max_valency, 
    const float decay,
    const unsigned kmax)
{
  vector<list<Edge>> k_best;
  list<list<long>> paths;
  list<list<long>> right_covers;
  vector<Edge> graph = buildGraph(
      left, right,
      ls, rs, 
      idf_right,
      pad);
  vector<Edge *> graph_refs(graph.size());
  for (unsigned i = 0; i < graph.size(); i++)
    graph_refs[i] = &graph[i];
  sort(graph_refs.begin(), graph_refs.begin() + graph_refs.size(), compareEdges);
  // printGraph(graph_refs);
  vector<Node> dag = buildDAGFromGraph(graph_refs, max_valency);
  // printDAG(dag);
  // DATA STRUCTURE FOR DECAY
  vector<list<unsigned>> map_right_to_edges(rs);
  for (unsigned i = 0; i < graph_refs.size(); i++)
    map_right_to_edges[graph_refs[i]->y].push_back(i);

  for (unsigned k = 0; k < kmax; k++)
  {
    // get best path at current decayed cost
    list<long> path = get_single_longest_path(dag);

    // decay
    for (const long &e : path)
      if (e <= (long)graph.size() && e > 0)
        for (const unsigned &i : map_right_to_edges[graph_refs[e - 1]->y])
          if (dag[i + 1].cost > numeric_limits<float>::epsilon() / decay)
          dag[i + 1].cost *= decay;

    // test right side redundancy
    list<long> right_cover;
    for (long &e : path)
      if (e <= (long)graph.size() && e != 0)
        right_cover.emplace_back(graph_refs[e - 1]->y);
    bool redundant = false;
    for (list<long> &other_cover : right_covers)
    {
      auto it = right_cover.begin();
      auto it_other = other_cover.begin();
      redundant = true;
      while (redundant && it != right_cover.end() && it_other != other_cover.end())
      {
        redundant = *it == *it_other;
        it = next(it);
        it_other = next(it_other);
      }
      if (redundant)
        break;
    }
    if (!redundant)
    {
      paths.emplace_back(path);
      right_covers.emplace_back(right_cover);
      list<Edge> cover;
      for (long &e : path)
        if (e <= (long)graph.size() && e != 0)
          cover.emplace_back(Edge(graph_refs[e - 1]->x, graph_refs[e - 1]->y, graph_refs[e - 1]->cost));
      k_best.push_back(cover);
    }
    else
      break;
  }
  return k_best;
}

inline
float compute_coverage_score(vector<vector<list<Edge>>> &covers, vector<unsigned> &choice, unsigned seq_len)
{
  vector<float> right_cover(seq_len, 0.f);
  for (unsigned n = 0; n < covers.size(); n++)
    if (covers[n].size() > 0)
      for (const Edge &e : covers[n][choice[n]])
      {
        right_cover[e.y] = e.cost;
      }
  float score = 0.f;
  for (const float &s : right_cover)
    score += s; 
  return score;
}

vector<unsigned> coverChoice(vector<vector<list<Edge>>> &covers, unsigned seq_len)
{
  vector<unsigned> best_choice(covers.size(), 0);
  float best_score = 0.f;
  vector<unsigned> current_choice(covers.size(), 0);

  unsigned current_explored = covers.size() - 1;
  float score = best_score;
  while (current_explored > 0 || current_choice[current_explored] < covers[current_explored].size())
  {
    current_explored = covers.size() - 1;

    // compute coverage
    score = compute_coverage_score(covers, current_choice, seq_len);
    // test if best score as far
    if (score > best_score)
    {
      best_score = score;
      for (unsigned j = 0; j < current_choice.size(); j++)
        best_choice[j] = current_choice[j];
    }
    // update current choice & current_explored
    current_choice[current_explored]++;
    while (current_explored > 0 && current_choice[current_explored] >= covers[current_explored].size())
    {
      current_choice[current_explored] = 0;
      current_explored--;
      current_choice[current_explored]++;
    }
  }
  return best_choice;
}

//////// GET SINGLE OP
void getOpsFromSingle(
    const long *s_i, const long *s_ref,
    const long s_i_len, const long s_ref_len,
    const float *idf_ref,
    const long n, const long k, const long max_valency, const float decay,
    long *del, long *ins, long *cmb,
    long *s_del, long *s_plh, long *s_cmb,
    const long pad, const long unk)
{
  const long seq_len = max(s_i_len, s_ref_len);

  vector<vector<list<Edge>>> covers = vector<vector<list<Edge>>>(n);
  for (long i = 0; i < n; ++i)
  {
    // get at most k alignment candidates between y_i and y
    covers[i] = get_k_best(
      &s_i[i * s_i_len],
      s_ref,
      s_i_len,
      s_ref_len,
      idf_ref,
      pad,
      max_valency, 
      decay,
      k);
  }
  // get best combination amongst
  vector<unsigned> choices = coverChoice(covers, seq_len);

  // fill output with results
  long j;
  for (long l = 0; (l < s_ref_len) && (s_ref[l] != pad); ++l)
  {
    s_cmb[l] = unk;
  }
  for (long i = 0; i < n; ++i)
  {
    // cout << i << endl;
    if (covers[i].size() == 0)
      continue;
    j = choices[i];
    long cpt_index = 0;
    long cpt_ins = -1;
    for (long l = 0; (l < s_ref_len) && (s_ref[l] != pad); ++l)
    {
      s_plh[i * seq_len + l] = unk;
    }

    list<Edge> &edges =  covers[i][j];
    for (const Edge &e : edges)
    {
      del[i * seq_len + e.x] = 1;
      if (cpt_index > 0)
      {
        ins[i * (seq_len - 1) + cpt_index - 1] = e.y - cpt_ins - 1;
      }
      cmb[i * seq_len + e.y] = 1;
      // cout << "s_del[" << i << "," << cpt_index << "] = s_i[" << i << "," << e.x << "] = " << s_i[i * seq_len + e.x] << endl;
      s_del[i * seq_len + cpt_index] = s_i[i * seq_len + e.x];
      s_plh[i * seq_len + e.y] = s_ref[e.y];
      s_cmb[e.y] = s_ref[e.y];

      cpt_ins = e.y;
      cpt_index++;
    }
  }
}

void getOpsFromBatch(
    const long *s_i,
    const long *s_ref,
    const long s_i_len,
    const long s_ref_len,
    const float *idf_ref,
    const long bsz,
    const long n,
    const long k,
    const long max_valency,
    const float decay,
    long *del,
    long *ins,
    long *cmb,
    long *s_del,
    long *s_plh,
    long *s_cmb,
    const long pad,
    const long unk)
{
  const long seq_len = max(s_i_len, s_ref_len);
  vector<thread> threads = vector<thread>();
  thread t;
  for (long b = 0; b < bsz; ++b)
  {
    t = thread(
        getOpsFromSingle,
        &s_i[b * n * s_i_len],
        &s_ref[b * s_ref_len],
        s_i_len,
        s_ref_len,
        &idf_ref[b * s_ref_len],
        n,
        k,
        max_valency,
        decay,
        &del[b * n * seq_len],
        &ins[b * n * (seq_len - 1)],
        &cmb[b * n * seq_len],
        &s_del[b * n * seq_len],
        &s_plh[b * n * seq_len],
        &s_cmb[b * seq_len],
        pad,
        unk);
    threads.push_back(move(t));
  }
  for (thread &t : threads)
  {
    t.join();
  }
}

int main()
{
  // cout << "Hello World!" << endl;
  const float decay = 0.0001f;
  // const vector<long> left_vec = {0, 1, 5, 4};
  // const vector<long> right_vec = {5, 0, 1};
  const unsigned n = 1;
  // const vector<long> left_vec = {9, 0, 0, 4, 1, 8,
  //                                9, 8, 10, 10, 10, 10,
  //                                9, 2, 1, 8, 10, 10};
  // const vector<long> right_vec = {9, 1, 0, 0, 4, 8, 10};
  // const vector<long> left_vec = {
  //   0,  6, 13,  7, 28,  7, 29,  5,  5,  6,  5,  5, 16,  5,  4,  4, 16,
  //          6, 29,  6,  4,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  //          1,  1,  1,  1,  1,  1,  1,
  //       0,  4,  5,  5,  5,  5,  6,  7,  5,  7,  4,  6,  4, 22,  7,  7,  7,
  //          5, 13,  7, 17,  4, 17,  4,  5,  2,  1,  1,  1,  1,  1,  1,  1,  1,
  //          1,  1,  1,  1,  1,  1,  1
  // };
  // const vector<long> right_vec = {
  //   0, 13,  6, 13, 28,  7, 29,  5,  7,  5,  5,  5,  7,  5, 16,  5,  4,  6,
  //         4, 29,  4,  5,  5, 14,  5, 18,  5,  6,  5,  7,  6,  7, 13,  7, 17, 17,
  //         5,  6,  5,  2,  1
  // };
  const vector<long> left_vec = {
          0, 19,  5,  4,  4, 22,  5,  6,  7,  6,  5, 10,  7,  6,  6, 15, 14,
           7,  4,  5,  7,  4,  6,  4, 23,  2,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
  };
  const vector<long> right_vec = {
    0, 19, 19,  5,  5,  4,  6,  4, 22,  6, 22,  5,  6,  5, 10,  7, 28,  6,
         28, 14, 16,  4,  6,  4,  4, 21,  5,  5, 27,  4,  4,  4, 23, 23, 14,  4,
         10,  4, 22, 26, 23,  5,  7,  7, 15,  5,  5,  2
  };

  // const vector<float> idf_right_vec = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f};
  const vector<float> idf_right_vec(right_vec.size(), 1.f);
  const long *left = &left_vec[0];
  const long *right = &right_vec[0];
  const float *idf_right = &idf_right_vec[0];

  const long ls = left_vec.size();
  const long rs = right_vec.size();
  const long seq_len = max(ls / n, rs);

  // cout << "LENGTH = " << n * seq_len << endl;

  long *del = (long *)calloc(n * seq_len, sizeof(long));
  long *ins = (long *)calloc(n * (seq_len - 1), sizeof(long));
  long *cmb = (long *)calloc(n * seq_len, sizeof(long));
  long *s_del = (long *)calloc(n * seq_len, sizeof(long));
  long *s_plh = (long *)calloc(n * seq_len, sizeof(long));
  long *s_cmb = (long *)calloc(n * seq_len, sizeof(long));

  const long pad = 1;
  const long unk = 3;
  const long max_valency = 10;
  const unsigned k = 10;

  const unsigned left_len = ls / n;

  getOpsFromSingle(
    left, right,
    left_len, rs,
    idf_right,
    n, k, max_valency, decay,
    del, ins, cmb,
    s_del, s_plh, s_cmb,
    pad, unk);

  cout << "y_plh:" << endl;
  for (unsigned i = 0; i < n; i++)
  {
    for (unsigned j = 0;  j < seq_len; j++)
    {
      cout << s_del[i * seq_len + j] << " ";
    }
    cout << endl;
  }

  // cout << "del:" << endl;
  // for (unsigned i = 0; i < n; i++)
  // {
  //   for (unsigned j = 0;  j < seq_len; j++)
  //   {
  //     cout << del[i * seq_len + j] << " ";
  //   }
  //   cout << endl;
  // }

  // cout << "plh:" << endl;
  // for (unsigned i = 0; i < n; i++)
  // {
  //   for (unsigned j = 0;  j < seq_len - 1; j++)
  //   {
  //     cout << ins[i * (seq_len - 1) + j] << " ";
  //   }
  //   cout << endl;
  // }

  // cout << "cmb:" << endl;
  // for (unsigned i = 0; i < n; i++)
  // {
  //   for (unsigned j = 0;  j < seq_len; j++)
  //   {
  //     cout << cmb[i * seq_len + j] << " ";
  //   }
  //   cout << endl;
  // }

  // vector<list<Edge>> k_best = get_k_best(left, right, ls, rs, pad, max_valency, decay, 3);

  free(del);
  free(ins);
  free(cmb);
  free(s_del);
  free(s_plh);
  free(s_cmb);

  return 0;
}


class EditOpsBatchIDF
{
public:
  const long bsz;
  const long n;
  const long s_i_len;
  const long s_ref_len;
  const long k;

  torch::Tensor del;
  torch::Tensor ins;
  torch::Tensor cmb;

  torch::Tensor s_del;
  torch::Tensor s_ins;
  torch::Tensor s_cmb;

  EditOpsBatchIDF(
      torch::Tensor s_i, torch::Tensor s_ref,
      torch::Tensor idf_ref,
      const long k_, const long max_valency_, const float decay,
      const long pad, const long unk) :
      bsz(s_i.size(0)), n(s_i.size(1)), s_i_len(s_i.size(2)), s_ref_len(s_ref.size(1)),
      k(k_)
  {
    const long max_valency = max_valency_ <= 0 ? LONG_MAX : max_valency_;
    long seq_len = max(s_i_len, s_ref_len);

    auto options = torch::TensorOptions()
                       .layout(s_i.layout())
                       .dtype(torch::kI64);

    // initialize ops
    del = torch::zeros({bsz, n, seq_len}, options);
    ins = torch::zeros({bsz, n, seq_len - 1}, options);
    cmb = torch::zeros({bsz, n, seq_len}, options);

    s_del = torch::full({bsz, n, seq_len}, pad, options);
    s_ins = torch::full({bsz, n, seq_len}, pad, options);
    s_cmb = torch::full({bsz, seq_len}, pad, options);

    getOpsFromBatch(
        s_i.data_ptr<long>(), s_ref.data_ptr<long>(),
        s_i_len, s_ref_len,
        idf_ref.data_ptr<float>(),
        bsz,
        n,
        k,
        max_valency,
        decay,
        del.data_ptr<long>(),
        ins.data_ptr<long>(),
        cmb.data_ptr<long>(),
        s_del.data_ptr<long>(),
        s_ins.data_ptr<long>(),
        s_cmb.data_ptr<long>(),
        pad, unk);
  };
  torch::Tensor getDel() { return del; };
  torch::Tensor getIns() { return ins; };
  torch::Tensor getCmb() { return cmb; };
  torch::Tensor getSDel() { return s_del; };
  torch::Tensor getSIns() { return s_ins; };
  torch::Tensor getSCmb() { return s_cmb; };
};

PYBIND11_MODULE(libnat3, m)
{
  py::class_<EditOpsBatchIDF>(m, "MultiLevEditOpsIDF")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, const long, const long, const float, const long, const long>())
      .def("get_del", &EditOpsBatchIDF::getDel)
      .def("get_ins", &EditOpsBatchIDF::getIns)
      .def("get_cmb", &EditOpsBatchIDF::getCmb)
      .def("get_s_del", &EditOpsBatchIDF::getSDel)
      .def("get_s_ins", &EditOpsBatchIDF::getSIns)
      .def("get_s_cmb", &EditOpsBatchIDF::getSCmb);
}
