package hmmlearn.base

import breeze.linalg._


object Utils {

  /**
    * Normalises the input array so that it sums to 1.
    * @param a Non-normalised input data.
    * @param axis Dimension along which normalization is performed (0 columns, 1 rows).
    * @return Normalised version of the input data.
    */
  def normalise(a: DenseMatrix[Double], axis: Int): DenseMatrix[Double] = {
    axis match {
      case 0 => {
        val aSum: DenseVector[Double] = sum(a, Axis._0).t
        a(*, ::) / aSum
      }
      case 1 => {
        val aSum: DenseVector[Double] = sum(a, Axis._1)
        a(::, *) / aSum
      }
    }
  }

  /**
    * Normalizes the input matrix so that the exponent of the sum is 1.
    * @param a Non-normalized input data.
    * @return Log normalised version of the input data.
    */
  def logNormalise(a: DenseMatrix[Double]): DenseMatrix[Double] = {
    val result = a.copy
    (0 to a.rows-1).foreach(row => {
      result(row, ::) -= softmax(a(row, ::))
    })
    result
  }

  def iterFromXLengths(nSamples: Int, lengths: Option[DenseVector[Int]]): Array[(Int, Int)] = {
    if (lengths.isEmpty) {
      Array((0, nSamples))
    } else {
      val end: DenseVector[Int] = accumulate(lengths.get)
      val start: DenseVector[Int] = end - lengths.get
      if (end(-1) > nSamples) {
        throw ValueError(s"More than $nSamples samples in lengths array")
      }
     start.data.zip(end.data)
    }
  }

}
