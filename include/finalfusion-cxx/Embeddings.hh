#ifndef FINALFUSION_CXX_EMBEDDINGS_HH
#define FINALFUSION_CXX_EMBEDDINGS_HH

#include <string>
#include <vector>

#include "finalfusion.h"

class EmbeddingsImpl;

/**
 * Embeddings.
 */
class Embeddings {
public:
  /**
   * Embeddings Constructor.
   *
   * @param filename path to embeddings.
   * @param mmap memmap embeddings.
   * @throws runtime_error if Embeddings could not be read.
   */
  Embeddings(std::string const &filename, bool mmap);

  /**
   * Method to load Embeddings from a fastText file.
   *
   * @param filename path to embeddings.
   * @throws runtime_error if Embeddings could not be read.
   */
  static Embeddings read_fasttext(std::string const &filename);

  /**
   * Method to load Embeddings from a text file.
   *
   * Read the word embeddings from a text stream. The text should contain one
   * word embedding per line in the following format:
   *
   * *word0 component_1 component_2 ... component_n*
   *
   * @param filename path to embeddings.
   * @throws runtime_error if Embeddings could not be read.
   */
  static Embeddings read_text(std::string const &filename);

  /**
   * Method to load Embeddings from a text file with dimensions.
   *
   * Read the word embeddings from a text stream. The text must contain
   * as the first line the shape of the embedding matrix:
   *
   * *vocab_size n_components*
   *
   * The remainder of the stream should contain one word embedding per line in
   * the following format:
   *
   * *word0 component_1 component_2 ... component_n*
   *
   * @param filename path to embeddings.
   * @throws runtime_error if Embeddings could not be read.
   */
  static Embeddings read_text_dims(std::string const &filename);

  /**
   * Method to load Embeddings from a word2vec binary file.
   *
   * Read the word embeddings from a file in word2vec binary format.
   *
   * @param filename path to embeddings.
   * @throws runtime_error if Embeddings could not be read.
   */
  static Embeddings read_word2vec(std::string const &filename);

  virtual ~Embeddings();

  /// Return embedding dimensionality.
  size_t dimensions();

  /**
   * Embedding lookup
   * @param word the query word
   * @return the embedding. Empty if none could be found.
   */
  std::vector<float> embedding(std::string const &word);

private:
  // Private constructor for different formats.
  Embeddings(EmbeddingsImpl *embeddings_impl);

  std::shared_ptr<EmbeddingsImpl> embeddings_impl_;
};

#endif // FINALFUSION_CXX_EMBEDDINGS_HH
