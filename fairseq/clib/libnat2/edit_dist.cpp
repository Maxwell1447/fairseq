// #include <torch/torch.h> // @manual=//caffe2:torch_extension
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string.h>
#include <utility>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <thread>
#include <limits.h>
#include "edit_dist.h"

using namespace std;
// namespace py = pybind11;

Edge::Edge(long x, long y) : x(x), y(y)
{
}

bool Edge::operator<(Edge const &e) const
{
  return (x + y) < (e.x + e.y);
}
bool Edge::operator<=(Edge const &e) const
{
  return (x + y) <= (e.x + e.y);
}
bool Edge::operator>(Edge const &e) const
{
  return (x + y) > (e.x + e.y);
}
bool Edge::operator>=(Edge const &e) const
{
  return (x + y) >= (e.x + e.y);
}
bool Edge::operator==(Edge const &e) const
{
  return (x == y) & (e.x == e.y);
}
void Edge::printEdge() const
{
  cout << "(" << x << ", " << y << "), ";
}

Node::Node(long index) : index(index)
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

Path::Path()
{
  path = {};
  completed = false;
}

Path::Path(Path p, long node, long s)
{
  path = list<long>(p.path);
  path.push_back(node);
  score = s;
  completed = false;
}

bool Path::operator<(Path const &p) const
{
  return score < p.score;
}
bool Path::operator<=(Path const &p) const
{
  return score <= p.score;
}
bool Path::operator>(Path const &p) const
{
  return score > p.score;
}
bool Path::operator>=(Path const &p) const
{
  return score >= p.score;
}
bool Path::operator==(Path const &p) const
{
  return score == p.score;
}
void Path::printPath() const
{
  for (long const &node : path)
  {
    cout << node << "<-";
  }
  cout << "\t\tscore = " << score;
  if (completed)
  {
    cout << "\t completed";
  }
  cout << "\n";
}

list<Edge> buildGraph(
    const long *left, const long *right,
    const long &size_left, const long &size_right, const long &pad)
{
  list<Edge> graph;
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
      Edge edge = Edge(i, j);
      graph.push_back(edge);
    }
  }
  return graph;
};

void printGraph(list<Edge> graph)
{
  for (Edge const &edge : graph)
  {
    edge.printEdge();
  }
  cout << "\n";
};

vector<Node> buildDAGFromGraph(vector<Edge> &graph, const long &max_valency)
{
  vector<Node> dag = vector<Node>(graph.size() + 2, Node(0));

  for (Edge const &edge : graph)
  {
    edge.printEdge();
  }
  cout << endl;

  // printGraph(graph); // print sorted graph of edges

  // initialize dag
  for (long i = 0; i < (long)graph.size() + 2; ++i)
  {
    Node node = Node(i);
    dag.at(i) = node;
  }

  // initialize with source
  for (long i = 0; (i < (long)graph.size()) && (i < max_valency); ++i)
  {
    dag.at(0).addNext(&dag.at(i + 1));
    dag.at(i + 1).addPrec(&dag.at(0));
    cout << "s" << "->" << i << endl;
  }

  // build DAG core
  long current_valency;
  for (long i = 0; i < (long)graph.size(); ++i)
  {
    current_valency = 0;
    for (long j = i + 1; (j < (long)graph.size()) && (current_valency < max_valency) && dag.at(i + 1).hasPrec(); ++j)
    {
      // cout << i << "," << j << ": " << current_valency << "  | ";
      if (
        (((int)graph.at(i).x - (int)graph.at(j).x) * ((int)graph.at(i).y - (int)graph.at(j).y) > 0)
        )
      {
        // cout << endl;
        current_valency++;
        dag.at(i + 1).addNext(&dag.at(j + 1));
        dag.at(j + 1).addPrec(&dag.at(i + 1));
        cout << i << "->" << j << endl;
      }
    }
  }

  // finalize with target
  for (long i = 0; i < (long)graph.size(); ++i)
  {
    if (!dag.at(i + 1).hasNext() && dag.at(i + 1).hasPrec())
    {
      dag.at(i + 1).addNext(&dag.at(graph.size() + 1));
      dag.at(graph.size() + 1).addPrec(&dag.at(i + 1));
      cout << i << "->" << "t" << endl;
    }
  }

  return dag;
};

void insertPair(long *best_scores, long *element, const long &k)
{
  long index = 1;
  long temp[2];

  if (best_scores[1] < element[1])
  {
    best_scores[0] = element[0];
    best_scores[1] = element[1];
  }
  else
  {
    return;
  }
  while ((index < k) && (element[1] > best_scores[index * 2 + 1]))
  {
    temp[0] = best_scores[index * 2];
    temp[1] = best_scores[index * 2 + 1];
    best_scores[index * 2] = best_scores[(index - 1) * 2];
    best_scores[index * 2 + 1] = best_scores[(index - 1) * 2 + 1];
    best_scores[(index - 1) * 2] = temp[0];
    best_scores[(index - 1) * 2 + 1] = temp[1];
    index++;
  }
}

void forwardKBest(vector<Node> &dag, const long &k, long *tableOfDist)
{
  for (long i = 1; i < (long)dag.size(); ++i)
  {
    for (list<Node *>::reverse_iterator it = dag.at(i).prec.rbegin(); it != dag.at(i).prec.rend(); ++it)
    {
      if (tableOfDist[i * k * 2 + 0 * 2 + 1] < (**it).index + 1)
      {
        long pair[2] = {
            (**it).index,
            tableOfDist[(**it).index * k * 2 + (k - 1) * 2 + 1] + 1};
        insertPair(&tableOfDist[i * k * 2], pair, k);
      }
    }
  }
}

vector<list<long>> backwardKBest(long *table, const long &k, const long &dag_size)
{
  vector<list<long>> paths(k, list<long>());
  list<Path> candidates = {Path()};
  vector<Path> current(k);

  for (long j = 0; j < k - 1; ++j)
  {
    current.at(j).path.push_back(0);
    current.at(j).score = 0;
  }
  current.at(k - 1).path.push_back(dag_size - 1);
  current.at(k - 1).score = table[(dag_size - 1) * k * 2 + (k - 1) * 2 + 1];

  bool candidatesRemaining = true;

  long i = dag_size - 1;
  for (long num_iter = 0; candidatesRemaining && (num_iter < dag_size); ++num_iter)
  {
    candidates = {};
    for (long j = 0; j < k; ++j)
    {
      i = current.at(j).path.back();
      if (current.at(j).path.back() > 0)
      {
        for (long m = 0; m < k; ++m)
        {
          if ((table[i * k * 2 + m * 2 + 0] + table[i * k * 2 + m * 2 + 1]) > 0)
          {
            candidates.push_back(Path(
                current.at(j),
                table[i * k * 2 + m * 2 + 0],
                table[i * k * 2 + m * 2 + 1] + num_iter));
            if (table[i * k * 2 + m * 2 + 0] == 0)
            {
              candidates.back().completed = true;
            }
          }
        }
      };
    }
    candidates.sort();
    while (((long)candidates.size() < k) && (candidates.size() > 0))
    {
      Path p = Path();
      p.score = 0;
      p.path = {0};
      candidates.push_front(p);
    }

    // clean current that have not finished yet
    i = k - 1;
    for (long m = k - 1; (m >= 0); --m)
    {
      if (current.at(m).completed)
      {
        swap(current.at(m), current.at(i));
        i--;
      }
      else
      {
        Path p = Path();
        p.score = 0;
        p.path = {0};
        current.at(m) = p;
      }
    }
    // insert candidates when they surpass current paths
    if (candidates.size() > 0)
    {
      candidatesRemaining = true;
      for (list<Path>::reverse_iterator it = candidates.rbegin(); it != candidates.rend(); ++it)
      {
        if ((*it).score >= current.at(0).score && (*it).score > 0)
        {
          current.at(0) = (*it);
          for (long m = 1; (m < k) && (current.at(m - 1) >= current.at(m)); ++m)
          {
            swap(current.at(m - 1), current.at(m));
          }
        }
      }
    }
    else
    {
      candidatesRemaining = false;
    }
  }

  for (long j = k; j > 0; --j)
  {
    for (
      list<long>::reverse_iterator it = current.at(j - 1).path.rbegin();
      it != current.at(j - 1).path.rend();
      it++
    )
    {
      paths.at(j - 1).push_back(*it - 1);
    }
    if (paths.at(j - 1).size() > 1)
    {
      paths.at(j - 1).pop_front();
      paths.at(j - 1).pop_back();
    }
    else
    {
      paths.at(j - 1) = list<long>();
      break;
    }
  }

  for (long j = k; j > 0; --j)
  {
    for (
      list<long>::reverse_iterator it = current.at(j - 1).path.rbegin();
      it != current.at(j - 1).path.rend();
      it++
    ) {
      cout << *it << ">";
    }
    cout << endl;
  }

  return paths;
}

vector<list<vector<long>>> graphToIndexation(vector<list<long>> &paths, vector<Edge> &graph)
{
  vector<list<vector<long>>> out = vector<list<vector<long>>>(
    paths.size(),
    list<vector<long>>()
  );
  // paths k x ?
  for (long j = 0; j < (long)paths.size(); ++j)
  {
    for (long const &i : paths.at(j))
    {
      out.at(j).push_back({graph.at(i).x, graph.at(i).y});
    }
  }

  return out;
};

vector<list<vector<long>>> kBestGraphs(list<Edge> graph, const long &k, const long &max_valency)
{
  vector<list<vector<long>>> k_best;

  vector<Edge> graph_vec;
  for (Edge &e : graph)
  {
    graph_vec.push_back(e);
  }

  vector<Node> dag = buildDAGFromGraph(graph_vec, max_valency);
  // long table[dag.size() * k * 2] = {0};
  long *table = new long[(long)dag.size() * k * 2];
  // for (long i = 0; i < (long)dag.size() * k * 2; i++) table[i] = 0;
  forwardKBest(dag, k, table);
  vector<list<long>> paths = backwardKBest(table, k, dag.size());
  k_best = graphToIndexation(paths, graph_vec);

  delete [] table;
  return k_best;
};

void indexationMask(vector<list<vector<long>>> indexation, long seq_len, bool *out)
{
  // indexation: k x num_edge x 2
  for (long j = 0; j < (long)indexation.size(); ++j)
  {
    for (vector<long> const &pair : indexation.at(j))
    {
      out[j * 2 * seq_len + pair.at(0)] = 1;
      out[j * 2 * seq_len + seq_len + pair.at(1)] = 1;
    }
  }
};

list<long> filterRedundancy(bool *masked_k_best, const long &k, const long &seq_len)
{
  list<long> filter = list<long>();
  // masked_k_best k x 2 x L

  for (long j = 0; j < k; ++j)
  {
    bool found_no_match = true;
    for (long m = j + 1; (m < k) && (found_no_match); ++m)
    {
      bool match = true;
      for (long i = 0; (i < seq_len) && (match); ++i)
      {
        match = (masked_k_best[j * 2 * seq_len + seq_len + i] || masked_k_best[m * 2 * seq_len + seq_len + i]) == masked_k_best[m * 2 * seq_len + seq_len + i];
      }
      found_no_match = !match;
    }
    if (found_no_match)
    {
      filter.push_back(j);
    }
  }

  return filter;
};

void recursiveCoverSearch(
    const long &dim, long &max_score,
    bool *max_cover, vector<long> &choices,
    const vector<list<long>> &filters,
    const bool *all_masks,
    long current_dim,
    list<long> current_choice,
    bool *current_cover,
    const long &k, const long &seq_len)
{
  if (current_dim == dim)
  {
    long score = 0;
    for (long i = 0; i < seq_len; ++i)
      score += (long)current_cover[i];
    if (score > max_score)
    {
      max_score = score;
      max_cover = current_cover;
      choices = vector<long>(current_choice.begin(), current_choice.end());
    }
  }
  else
  {
    for (long const &m : filters.at(current_dim))
    {
      list<long> new_choice = list<long>(current_choice);
      new_choice.push_back(m);
      for (long i = 0; i < seq_len; ++i)
      {
        current_cover[i] = (current_cover[i] ||
                            all_masks[current_dim * k * 2 * seq_len + m * 2 * seq_len + 1 * seq_len + i]);
      }
      recursiveCoverSearch(
        dim, max_score,
        max_cover, choices,
        filters, all_masks,
        current_dim + 1,
        new_choice,
        current_cover,
        k, seq_len
      );
    }
  }
}

void getOpsFromSingle(
    const long *s_i, const long *s_ref,
    const long s_i_len, const long s_ref_len,
    const long &n, const long &k, const long &max_valency,
    long *del, long *ins, long *cmb,
    long *s_del, long *s_plh, long *s_cmb,
    const long &pad, const long &unk)
{
  const long seq_len = max(s_i_len, s_ref_len);
  bool *all_masked = new bool[n * k * 2 * seq_len];
  vector<vector<list<vector<long>>>> all_indexations = vector<vector<list<vector<long>>>>(n);
  vector<list<long>> filters = vector<list<long>>(n);
  for (long i = 0; i < n; ++i)
  {
    list<Edge> graph = buildGraph(&s_i[i * s_i_len], s_ref, s_i_len, s_ref_len, pad);

    graph.sort();

    all_indexations.at(i) = kBestGraphs(graph, k, max_valency);

    bool *out = new bool[k * 2 * seq_len];
    indexationMask(all_indexations.at(i), seq_len, out);

    for (long m = 0; m < (k * 2 * seq_len); ++m)
    {
      all_masked[i * k * 2 * seq_len + m] = out[m];
      // cout << " " << out[m];
    }
    // for (long kk = 0; kk < k; kk++) {
    //   cout << kk << "\t";
    //   for (long ll = 0; ll < seq_len; ll++) {
    //     // if (out[kk * 2 * seq_len + 1 * seq_len + ll]) {
    //     //   cout << "1";
    //     // }
    //     // else {
    //     //   cout << "0";
    //     // }
    //     cout << out[kk * 2 * seq_len + 1 * seq_len + ll];
    //     cout << "";
    //   }
    //   cout << endl;
    // }
    // cout << endl;

    filters.at(i) = filterRedundancy(&all_masked[i * k * 2 * seq_len], k, seq_len);

    cout << "filtered : ";
    for (auto const &m : filters.at(i)) {
      cout << m << ", ";
    }
    cout << endl;

    delete [] out;
  }

  long max_score = 0;
  bool *max_cover = new bool[seq_len];
  bool *current_cover = new bool[seq_len];
  vector<long> choices = vector<long>(n);
  recursiveCoverSearch(
    n, max_score, max_cover,
    choices, filters,
    all_masked,
    0,
    list<long>(),
    current_cover,
    k, seq_len
  );

  long j;
  for (long l = 0; (l < s_ref_len) && (s_ref[l] != pad); ++l)
  {
    s_cmb[l] = unk;
  }
  for (long i = 0; i < n; ++i)
  {
    j = choices.at(i);
    cout << "choice " << i << " = " << j << endl;
    long cpt_index = 0;
    long cpt_ins = -1;
    for (long l = 0; (l < s_ref_len) && (s_ref[l] != pad); ++l)
    {
      s_plh[i * seq_len + l] = unk;
    }
    for (vector<long> const &idx : all_indexations.at(i).at(j))
    {
      del[i * seq_len + idx.at(0)] = 1;
      if (cpt_index > 0)
      {
        ins[i * (seq_len - 1) + cpt_index - 1] = idx.at(1) - cpt_ins - 1;
      }
      cmb[i * seq_len + idx.at(1)] = 1;

      s_del[i * seq_len + cpt_index] = s_i[i * seq_len + idx.at(0)];
      s_plh[i * seq_len + idx.at(1)] = s_ref[idx.at(1)];
      s_cmb[idx.at(1)] = s_ref[idx.at(1)];

      cpt_ins = idx.at(1);
      cpt_index++;
    }
  }
  delete [] all_masked;
  delete [] max_cover;
  delete [] current_cover;
}


int main() {

  cout << "Hello World!";

  return 0;
}

// void getOpsFromBatch(
//     const long *s_i,
//     const long *s_ref,
//     const long &s_i_len,
//     const long &s_ref_len,
//     const long &bsz,
//     const long &n,
//     const long &k,
//     const long &max_valency,
//     long *del,
//     long *ins,
//     long *cmb,
//     long *s_del,
//     long *s_plh,
//     long *s_cmb,
//     const long &pad,
//     const long &unk)
// {
//   long seq_len = max(s_i_len, s_ref_len);
//   vector<thread> threads = vector<thread>();
//   thread t;
//   for (long b = 0; b < bsz; ++b)
//   {
//     t = thread(
//         getOpsFromSingle,
//         &s_i[b * n * s_i_len],
//         &s_ref[b * s_ref_len],
//         s_i_len,
//         s_ref_len,
//         n,
//         k,
//         max_valency,
//         &del[b * n * seq_len],
//         &ins[b * n * (seq_len - 1)],
//         &cmb[b * n * seq_len],
//         &s_del[b * n * seq_len],
//         &s_plh[b * n * seq_len],
//         &s_cmb[b * seq_len],
//         pad,
//         unk);
//     threads.push_back(move(t));
//   }
//   for (thread &t : threads)
//   {
//     t.join();
//   }
// }

// class EditOpsBatch
// {
// public:
//   long bsz;
//   long n;
//   long s_i_len;
//   long s_ref_len;
//   long k;

//   torch::Tensor del;
//   torch::Tensor ins;
//   torch::Tensor cmb;

//   torch::Tensor s_del;
//   torch::Tensor s_ins;
//   torch::Tensor s_cmb;

//   EditOpsBatch(){};
//   EditOpsBatch(
//       torch::Tensor s_i, torch::Tensor s_ref,
//       long k_, long max_valency, long pad, long unk)
//   {
//     bsz = s_i.size(0);
//     n = s_i.size(1);
//     k = k_;
//     if (max_valency <= 0) {
//       max_valency = LONG_MAX;
//     }
//     s_i_len = s_i.size(2);
//     s_ref_len = s_ref.size(1);
//     long seq_len = max(s_i_len, s_ref_len);

//     auto options = torch::TensorOptions()
//                        .dtype(torch::kI64);

//     // initialize ops
//     del = torch::zeros({bsz, n, seq_len}, options);
//     ins = torch::zeros({bsz, n, seq_len - 1}, options);
//     cmb = torch::zeros({bsz, n, seq_len}, options);

//     s_del = torch::full({bsz, n, seq_len}, pad, options);
//     s_ins = torch::full({bsz, n, seq_len}, pad, options);
//     s_cmb = torch::full({bsz, seq_len}, pad, options);

//     getOpsFromBatch(
//         s_i.data_ptr<long>(), s_ref.data_ptr<long>(),
//         s_i_len, s_ref_len,
//         bsz,
//         n,
//         k,
//         max_valency,
//         del.data_ptr<long>(),
//         ins.data_ptr<long>(),
//         cmb.data_ptr<long>(),
//         s_del.data_ptr<long>(),
//         s_ins.data_ptr<long>(),
//         s_cmb.data_ptr<long>(),
//         pad, unk);
//   };
//   torch::Tensor getDel() { return del; };
//   torch::Tensor getIns() { return ins; };
//   torch::Tensor getCmb() { return cmb; };
//   torch::Tensor getSDel() { return s_del; };
//   torch::Tensor getSIns() { return s_ins; };
//   torch::Tensor getSCmb() { return s_cmb; };
// };

// PYBIND11_MODULE(libnat2, m)
// {
//   py::class_<EditOpsBatch>(m, "MultiLevEditOps")
//       .def(py::init<torch::Tensor, torch::Tensor, long, long, long, long>())
//       .def(py::init<>())
//       .def("get_del", &EditOpsBatch::getDel)
//       .def("get_ins", &EditOpsBatch::getIns)
//       .def("get_cmb", &EditOpsBatch::getCmb)
//       .def("get_s_del", &EditOpsBatch::getSDel)
//       .def("get_s_ins", &EditOpsBatch::getSIns)
//       .def("get_s_cmb", &EditOpsBatch::getSCmb);
// }
