package hmmlearn.base

import breeze.linalg._
import breeze.numerics.{exp, log}
import hmmlearn.base.BaseHMM.Stats

import scala.util.Random

object BaseHMM {

  // nobs, start, trans, obs
  type Stats = (Int, DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double])

  def doForwardPass(
                     frameLogProb: DenseMatrix[Double],
                     startProb: DenseVector[Double],
                     transMat: DenseMatrix[Double]
                   ): (Double, DenseMatrix[Double]) = {
    val nSamples = frameLogProb.rows
    val nComponents = frameLogProb.cols
    val fwdLattice = Optimisation.forward(nSamples, nComponents, log(startProb), log(transMat), frameLogProb)
    (softmax(fwdLattice(-1, ::).t), fwdLattice)
  }

  def doBackwardPass(
                      frameLogProb: DenseMatrix[Double],
                      startProb: DenseVector[Double],
                      transMat: DenseMatrix[Double]
                    ): DenseMatrix[Double] = {
    val nSamples = frameLogProb.rows
    val nComponents = frameLogProb.cols
    Optimisation.backward(nSamples, nComponents, log(startProb), log(transMat), frameLogProb)
  }

  def computePosteriors(
                         fwdLattice: DenseMatrix[Double],
                         bwdLattice: DenseMatrix[Double]
                       ): DenseMatrix[Double] = {
    val logGamma = fwdLattice + bwdLattice
    exp(Utils.logNormalise(logGamma))
  }

}

abstract class BaseHMM[T](
                           nComponents: Int,
                           startProbPrior: Option[DenseVector[Double]] = None,
                           transMatPrior: Option[DenseMatrix[Double]] = None,
                           algorithm: String = "viterbi",
                           randomState: Option[Any] = None,
                           nIterations: Int = 10,
                           tolerance: Double = 1e-2,
                           params: String = "st",
                           initParams: String = ""
                         ) {

  val startProb: DenseVector[Double] = startProbPrior.getOrElse(DenseVector.zeros[Double](nComponents))
  val transMat: DenseMatrix[Double] = transMatPrior.getOrElse(DenseMatrix.zeros[Double](nComponents, nComponents))

  // TODO: Check the matrices are ok

  def scoreSamples(X: DenseVector[T], lengths: Option[DenseVector[Int]] = None): (Double, DenseMatrix[Double]) = {
    // TODO: check_is_fitted
    // TODO: check_array
    val nSamples: Int = X.length
    var logProb = 0.0
    val posteriors = DenseMatrix.zeros[Double](nSamples, nComponents)
    Utils.iterFromXLengths(X.length, lengths).foreach(f => {
      val (start, end) = f
      val frameLogProb: DenseMatrix[Double] = computeLogLikelihood(X(start until end))
      val (logProbSample, fwdLattice) = BaseHMM.doForwardPass(frameLogProb, startProb, transMat)
      logProb += logProbSample
      val bwdLattice: DenseMatrix[Double] = BaseHMM.doBackwardPass(frameLogProb, startProb, transMat)
      posteriors(start until end, ::) := BaseHMM.computePosteriors(fwdLattice, bwdLattice)
    })
    (logProb, posteriors)
  }

  def generateSampleFromState(state: Int, randomState: Random): DenseVector[Double] = ???

  def initialiseSufficientStatistics: Stats = ???

  def accumulateSufficientStatistics(
                                      stats: Stats,
                                      X: DenseVector[T],
                                      frameLogProb: DenseMatrix[Double],
                                      posteriors: DenseMatrix[Double],
                                      fwdLattice: DenseMatrix[Double],
                                      bwdLattice: DenseMatrix[Double]
                                    ): Stats = {
    val nobs = stats._1 + 1
    val start: DenseVector[Double] = if (params.contains("s")) stats._2 + posteriors(0, ::).t else stats._2
    val trans: DenseMatrix[Double] = if (params.contains("t")) {
      val nSamples = frameLogProb.rows
      val nComponents = frameLogProb.cols
      if (nSamples <= 1) {
        stats._3
      } else {
        val logXiSum = Optimisation.computeLogXiSum(
          nSamples, nComponents, fwdLattice, log(transMat), bwdLattice, frameLogProb, Some(Double.NegativeInfinity)
        )
        stats._3 + exp(logXiSum)
      }
    } else {
      stats._3
    }
    new Stats(nobs, start, trans, stats._4)
  }

  def computeLogLikelihood(X: DenseVector[T]): DenseMatrix[Double] = ???

}
