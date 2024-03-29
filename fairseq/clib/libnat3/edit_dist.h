#pragma once

#include <cstdint>
#include <list>
#include <vector>
#include <iostream>
#include <utility>
#include <torch/extension.h>

#ifndef MLEVT_IDF_H
#define MLEVT_IDF_H

class Edge {
public:
  Edge(const long x, const long y, const float=1.f);
  const long x;
  const long y;
  const float cost;
  bool operator<(Edge const & e) const;
  bool operator==(Edge const & e) const;
  void printEdge() const;
};

class Node {
public:
  Node(const long index, const float cost=1.f);
  const long index;
  float cost;
  float current_length;
  std::list<Node*> next;
  std::list<Node*> prec;
  void addNext(Node* node);
  bool hasNext();
  void addPrec(Node* node);
  bool hasPrec();
};

#endif
