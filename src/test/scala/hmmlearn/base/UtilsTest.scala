package hmmlearn.base

import breeze.linalg._
import org.scalatest.{FlatSpec, Matchers}


class UtilsTest extends FlatSpec with Matchers {

  "normalise" should "compute the right result normalising a matrix along the first axis" in {
    val m: DenseMatrix[Double] = DenseMatrix((10.0, 15.0, 25.0), (15.0, 40.0, 25.0))
    val norm = Utils.normalise(m, 0)
    norm shouldBe DenseMatrix((0.4, 15.0/55, 0.5), (0.6, 40.0/55, 0.5))
  }

  it should "compute the right result normalising a matrix along the second axis" in {
    val m: DenseMatrix[Double] = DenseMatrix((10.0, 15.0, 25.0), (15.0, 40.0, 25.0))
    val norm = Utils.normalise(m, 1)
    norm shouldBe DenseMatrix((0.2, 0.3, 0.5), (0.1875, 0.5, 0.3125))
  }

  "logNormalise" should "compute the right log normalised matrix" in {
    val m = DenseMatrix((1.0, 2.0, 3.0), (9.0, 5.5, 6.1))
    val logNormalisedM = Utils.logNormalise(m)
    logNormalisedM shouldBe DenseMatrix(
      (-2.4076059644443806, -1.4076059644443806, -0.4076059644443806),
      (-0.08178328750402208, -3.581783287504022, -2.9817832875040224)
    )
  }

  "iterFromXLengths" should "compute the right results with no lengths" in {
    val indexes = Utils.iterFromXLengths(10, None)
    indexes shouldBe Array((0, 10))
  }

  it should "compute the right results with some lengths" in {
    val lengths = DenseVector(3, 4, 2, 1)
    val indexes = Utils.iterFromXLengths(10, Some(lengths))
    indexes shouldBe Array((0, 3), (3, 7), (7, 9), (9, 10))
  }

  it should "raise an exception if the sum of the lengths is greater than the length of the matrix" in {
    val lengths = DenseVector(3, 4, 2, 2)
    assertThrows[ValueError] {
      Utils.iterFromXLengths(10, Some(lengths))
    }
  }

}
