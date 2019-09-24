#include <memory>
#include <vector>

#include "finalfusion-cxx/Embeddings.hh"
#include "finalfusion.h"

int main() {
  try {
    std::unique_ptr<Embeddings> embeddings = std::unique_ptr<Embeddings>(
        new Embeddings(Embeddings::read_word2vec("testdata/test.w2v")));
    std::vector<float> embedding = embeddings->embedding("Berlin");
    if (embedding.empty()) {
      return 1;
    }

    embedding = embeddings->embedding("Tübingen");
    if (!embedding.empty()) {
      return 1;
    }

    return 0;
  } catch (std::exception &e) {
    return 1;
  }
}