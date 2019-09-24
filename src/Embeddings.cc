#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "finalfusion-cxx/Embeddings.hh"
#include "finalfusion.h"

/**
 * Enum to distinguish formats other than finalfusion.
 */
enum class NonFifuFormat { FastText, Text, TextDims, Word2Vec };

void delete_ff_embeddings(ff_embeddings_t *ptr) {
  ff_free_embeddings((ff_embeddings)ptr);
}

class EmbeddingsImpl {
public:
  EmbeddingsImpl(std::string const &filename, bool mmap)
      : d_inner(nullptr, delete_ff_embeddings) {

    if (mmap) {
      d_inner.reset(ff_mmap_embeddings(filename.c_str()));
    } else {
      d_inner.reset(ff_read_embeddings(filename.c_str()));
    }

    if (d_inner == nullptr) {
      throw std::runtime_error(std::string(ff_error()));
    }
  }

  EmbeddingsImpl(std::string const &filename, NonFifuFormat const format)
      : d_inner(nullptr, delete_ff_embeddings) {
    switch (format) {
    case NonFifuFormat::FastText:
      d_inner.reset(ff_read_fasttext(filename.c_str()));
      break;
    case NonFifuFormat::Text:
      d_inner.reset(ff_read_text(filename.c_str()));
      break;
    case NonFifuFormat::TextDims:
      d_inner.reset(ff_read_text_dims(filename.c_str()));
      break;
    case NonFifuFormat::Word2Vec:
      d_inner.reset(ff_read_word2vec(filename.c_str()));
      break;
    default:
      throw std::runtime_error("Unhandled embedding format.");
    }

    if (d_inner == nullptr) {
      throw std::runtime_error(std::string(ff_error()));
    }
  }

  ~EmbeddingsImpl() = default;

  size_t dimensions() { return ff_embeddings_dims(d_inner.get()); }

  std::vector<float> embedding(std::string const &word) {
    float *raw_embedding = ff_embedding_lookup(d_inner.get(), word.c_str());
    if (raw_embedding == nullptr) {
      return std::vector<float>();
    }

    size_t dims = dimensions();
    std::vector<float> embedding(raw_embedding, raw_embedding + dims);
    free(raw_embedding);
    return embedding;
  }

  std::unique_ptr<ff_embeddings_t, decltype(&delete_ff_embeddings)> d_inner;
};

Embeddings::Embeddings(std::string const &filename, bool mmap)
    : embeddings_impl_(std::shared_ptr<EmbeddingsImpl>(
          new EmbeddingsImpl(filename, mmap))){};

Embeddings::Embeddings(EmbeddingsImpl *embeddings_impl)
    : embeddings_impl_(embeddings_impl){};

Embeddings Embeddings::read_fasttext(std::string const &filename) {
  return Embeddings(new EmbeddingsImpl(filename, NonFifuFormat::FastText));
};

Embeddings Embeddings::read_text(std::string const &filename) {
  return Embeddings(new EmbeddingsImpl(filename, NonFifuFormat::Text));
};

Embeddings Embeddings::read_text_dims(std::string const &filename) {
  return Embeddings(new EmbeddingsImpl(filename, NonFifuFormat::TextDims));
};

Embeddings Embeddings::read_word2vec(std::string const &filename) {
  return Embeddings(new EmbeddingsImpl(filename, NonFifuFormat::Word2Vec));
};

Embeddings::~Embeddings() = default;

std::vector<float> Embeddings::embedding(std::string const &word) {
  return embeddings_impl_->embedding(word);
}

size_t Embeddings::dimensions() { return embeddings_impl_->dimensions(); }
