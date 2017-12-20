package hmmlearn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import hmmlearn.base.BaseHMM
import hmmlearn.base.BaseHMM.Stats


class MultinomialHMM(
                      nComponents: Int,
                      nFeatures: Int,
                      startProbPrior: Option[DenseVector[Double]] = None,
                      transMatPrior: Option[DenseMatrix[Double]] = None,
                      emissionProbPrior: Option[DenseMatrix[Double]] = None,
                      algorithm: String = "viterbi",
                      randomState: Option[Any] = None,
                      nIterations: Int = 10,
                      tolerance: Double = 1e-2,
                      params: String = "ste",
                      initParams: String = ""
                    ) extends BaseHMM[Int] (
  nComponents,
  startProbPrior,
  transMatPrior,
  algorithm,
  randomState,
  nIterations,
  tolerance,
  params,
  initParams
){

  val emissionProb: DenseMatrix[Double] = emissionProbPrior.getOrElse(DenseMatrix.zeros[Double](nComponents, nFeatures))

  override def computeLogLikelihood(X: DenseVector[Int]): DenseMatrix[Double] = {
    val logEmission: DenseMatrix[Double] = log(emissionProb)
    val indexedSeq: IndexedSeq[Int] = IndexedSeq(X.toArray:_*)
    logEmission(::, indexedSeq).toDenseMatrix.t
  }

  override def initialiseSufficientStatistics: Stats = new Stats(
    0,
    DenseVector.zeros[Double](nComponents),
    DenseMatrix.zeros[Double](nComponents, nComponents),
    DenseMatrix.zeros[Double](nComponents, nFeatures)
  )

  override def accumulateSufficientStatistics(
                                               stats: Stats,
                                               X: DenseVector[Int],
                                               frameLogProb: DenseMatrix[Double],
                                               posteriors: DenseMatrix[Double],
                                               fwdLattice: DenseMatrix[Double],
                                               bwdLattice: DenseMatrix[Double]
                                             ): Stats = {

    val results: Stats = super.accumulateSufficientStatistics(stats, X, frameLogProb, posteriors, fwdLattice, bwdLattice)
    if (params.contains("e")) {
      for {
        i <- 0 until X.length
      } yield {
        results._4(::, X(i)) := results._4(::, X(i)) + posteriors(i, ::).t
      }
    }
    results
  }

  """
    |    def _do_mstep(self, stats):
    |        super(MultinomialHMM, self)._do_mstep(stats)
    |        if 'e' in self.params:
    |            self.emissionprob_ = (stats['obs']
    |                                  / stats['obs'].sum(axis=1)[:, np.newaxis])
  """.stripMargin

}
