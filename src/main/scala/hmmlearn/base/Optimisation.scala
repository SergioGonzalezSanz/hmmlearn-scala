package hmmlearn.base

import breeze.linalg._
import breeze.numerics._


object Optimisation {

  def forward(
               nSamples: Int,
               nComponents: Int,
               logStartProb: DenseVector[Double],
               logTransMat: DenseMatrix[Double],
               frameLogProb: DenseMatrix[Double]
             ): DenseMatrix[Double] = {
    val workBuffer: DenseVector[Double] = DenseVector.zeros[Double](nComponents)
    val fwdLattice: DenseMatrix[Double] = DenseMatrix.zeros[Double](nSamples, nComponents)
    (0 until nComponents).foreach(i => {fwdLattice(0, i) = logStartProb(i) + frameLogProb(0, i)})
    for {
      t <- 1 until nSamples
      j <- 0 until nComponents
    } yield {
      for {
        i <- 0 until nComponents
      } yield {
        workBuffer(i) = fwdLattice(t - 1, i) + logTransMat(i, j)
      }
      fwdLattice(t, j) = softmax(workBuffer) + frameLogProb(t, j)
    }
    fwdLattice
  }

  def backward(
                nSamples: Int,
                nComponents: Int,
                logStartProb: DenseVector[Double],
                logTransMat: DenseMatrix[Double],
                frameLogProb: DenseMatrix[Double]
              ): DenseMatrix[Double] = {
    val workBuffer: DenseVector[Double] = DenseVector.zeros[Double](nComponents)
    val bwdLattice: DenseMatrix[Double] = DenseMatrix.zeros[Double](nSamples, nComponents)
    for {
      t <- (nSamples - 2) until -1 by -1
      i <- 0 until nComponents
    } yield {
      for {
        j <- 0 until nComponents
      } yield {
        workBuffer(j) = logTransMat(i, j) + frameLogProb(t + 1, j) + bwdLattice(t + 1, j)
      }
      bwdLattice(t, i) = softmax(workBuffer)
    }
    bwdLattice
  }

  def computeLogXiSum(
                       nSamples: Int,
                       nComponents: Int,
                       fwdLattice: DenseMatrix[Double],
                       logTransMat: DenseMatrix[Double],
                       bwdLattice: DenseMatrix[Double],
                       frameLogProb: DenseMatrix[Double],
                       startValue: Option[Double] = None
                     ): DenseMatrix[Double] = {
    val workBuffer: DenseMatrix[Double] = DenseMatrix.fill(nComponents, nComponents){Double.NegativeInfinity}
    val logXiSum: DenseMatrix[Double] = startValue match {
      case Some(value) => DenseMatrix.ones[Double](nComponents, nComponents) * value
      case None => DenseMatrix.zeros[Double](nComponents, nComponents)
    }
    val logProb = softmax(fwdLattice(-1, ::))

    for {
      t <- 0 until (nSamples - 1)
    } yield {
      for {
        i <- 0 until nComponents
        j <- 0 until nComponents
      } yield {
        workBuffer(i, j) = fwdLattice(t, i) + logTransMat(i, j) + frameLogProb(t + 1, j) + bwdLattice(t + 1, j) - logProb
      }
      for {
        i <- 0 until nComponents
        j <- 0 until nComponents
      } yield {
        logXiSum(i, j) = logAddExp(logXiSum(i, j), workBuffer(i, j))
      }
    }
    logXiSum
  }

  def logAddExp(a: Double, b: Double): Double = {
    (a, b) match {
      case (Double.NegativeInfinity, valueB) => valueB
      case (valueA, Double.NegativeInfinity) => valueA
      case (valueA, valueB) => max(a, b) + log1p(exp(-abs(valueA - valueB)))
    }
  }

  def viterbi(
               nSamples: Int,
               nComponents: Int,
               logStartProb: DenseVector[Double],
               logTransMat: DenseMatrix[Double],
               frameLogProb: DenseMatrix[Double]
             ): (DenseVector[Int], Double) = {
    val stateSequence = DenseVector.zeros[Int](nSamples)
    val viterbiLattice = DenseMatrix.zeros[Double](nSamples, nComponents)
    val workBuffer = DenseVector.zeros[Double](nComponents)
    for {
      i <- 0 until nComponents
    } yield {
      viterbiLattice(0, i) = logStartProb(i) + frameLogProb(0, i)
    }

    // Induction
    for {
      t <- 1 until nSamples
      i <- 0 until nComponents
    } yield {
      for {
        j <- 0 until nComponents
      } yield {
        workBuffer(j) = logTransMat(j, i) + viterbiLattice(t - 1, j)
      }
      viterbiLattice(t, i) = max(workBuffer) + frameLogProb(t, i)
    }

    // Observation traceback
    var whereFrom = argmax(viterbiLattice(-1, ::))
    stateSequence(nSamples - 1) = whereFrom
    val logProb = viterbiLattice(-1, whereFrom)
    for {
      t <- (nSamples - 2) until -1 by -1
    } yield {
      for {
        i <- 0 until nComponents
      } yield {
        workBuffer(i) = viterbiLattice(t, i) + logTransMat(i, whereFrom)
      }
      whereFrom = argmax(workBuffer)
      stateSequence(t) = whereFrom
    }
    (stateSequence, logProb)
  }

}
