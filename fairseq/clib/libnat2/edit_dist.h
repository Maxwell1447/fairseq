#pragma once

#include <cstdint>
#include <list>
#include <vector>
#include <iostream>
#include <utility>
// #include <torch/extension.h>

#ifndef GRAPH_LEV_H
#define GRAPH_LEV_H

class Edge {
public:
  Edge(long x, long y);
  long x;
  long y;
  bool operator<(Edge const & e) const;
  bool operator<=(Edge const & e) const;
  bool operator>(Edge const & e) const;
  bool operator>=(Edge const & e) const;
  bool operator==(Edge const & e) const;
  void printEdge() const;
};

class Node {
public:
  // Node();
  Node(long index);
  long index;
  std::list<Node*> next;
  std::list<Node*> prec;
  void addNext(Node* node);
  bool hasNext();
  void addPrec(Node* node);
  bool hasPrec();
};

class Path {
public:
  Path();
  Path(Path p, long node, long s);
  std::list<long> path;
  long score;
  bool completed;
  bool operator<(Path const & p) const;
  bool operator<=(Path const & p) const;
  bool operator>(Path const & p) const;
  bool operator>=(Path const & p) const;
  bool operator==(Path const & p) const;
  void printPath() const;
};

#endif
