package hmmlearn.base

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import hmmlearn.MultinomialHMM
import org.scalatest.{FlatSpec, Matchers}


class MutinomialHMMTest extends FlatSpec with Matchers {

  "computeLogLikelihood" should "produce the right results" in {
    val X = DenseVector(1, 0, 3, 5, 2, 4, 1)
    val emissionProb = DenseMatrix(
      (0.5, 0.1, 0.1, 0.1, 0.1, 0.1),
      (0.2, 0.2, 0.1, 0.05, 0.0, 0.45)
    )
    val model = new MultinomialHMM(
      nComponents = 4,
      nFeatures = 10,
      emissionProbPrior = Some(emissionProb)
    )

    val logLikelihood = model.computeLogLikelihood(X)
    logLikelihood shouldBe DenseMatrix(
      (-2.3025850929940455, -1.6094379124341003),
      (-0.6931471805599453, -1.6094379124341003),
      (-2.3025850929940455, -2.995732273553991),
      (-2.3025850929940455, -0.7985076962177716),
      (-2.3025850929940455, -2.3025850929940455),
      (-2.3025850929940455, Double.NegativeInfinity),
      (-2.3025850929940455, -1.6094379124341003)
    )
  }

  "initialiseSufficientStatistics" should "return an empty stats tuple" in {
    val nComponents = 10
    val nFeatures = 30
    val model = new MultinomialHMM(nComponents, nFeatures)
    val stats = model.initialiseSufficientStatistics
    stats._1 shouldBe 0
    stats._2 shouldBe DenseVector.zeros[Double](nComponents)
    stats._3 shouldBe DenseMatrix.zeros[Double](nComponents, nComponents)
    stats._4 shouldBe DenseMatrix.zeros[Double](nComponents, nFeatures)
  }

  "accumulateSufficientStatistics" should "produce the right results with 'ste' in params" in {
    val nComponents: Int = 4
    val nFeatures: Int = 5
    val startProb: Option[DenseVector[Double]] = Some(DenseVector(0.2, 0.05, 0.6, 0.15))
    val transMat: Option[DenseMatrix[Double]] = Some(DenseMatrix(
      (0.8, 0.1, 0.05, 0.05),
      (0.1, 0.7, 0.15, 0.05),
      (0.05, 0.1, 0.6, 0.25),
      (0.1, 0.05, 0.1, 0.75)
    ))
    val model = new MultinomialHMM(
      nComponents = nComponents,
      nFeatures = nFeatures,
      startProbPrior = startProb,
      transMatPrior = transMat
    )
    val stats = new BaseHMM.Stats(
      0,
      DenseVector.zeros[Double](nComponents),
      DenseMatrix.zeros[Double](nComponents, nComponents),
      DenseMatrix.zeros[Double](nComponents, nFeatures)
    )
    val X = DenseVector[Int](1, 4, 3)
    val frameLogProb = log(DenseMatrix(
      (0.05, 0.01, 0.90, 0.04),
      (0.85, 0.02, 0.03, 0.10),
      (0.12, 0.03, 0.80, 0.05)
    ))
    val (loss, fwdLattice) = BaseHMM.doForwardPass(frameLogProb, startProb.get, transMat.get)
    val bwdLattice = BaseHMM.doBackwardPass(frameLogProb, startProb.get, transMat.get)
    val posteriors = BaseHMM.computePosteriors(fwdLattice, bwdLattice)
    val results = model.accumulateSufficientStatistics(stats, X, frameLogProb, posteriors, fwdLattice, bwdLattice)
    results._1 shouldBe 1
    results._2 shouldBe DenseVector(0.0875428779442073, 7.649747594083709E-4, 0.8990820470485451, 0.012610100247838865)
    results._3 shouldBe DenseMatrix(
      (0.3460338435828014, 0.008403828460401856, 0.10901692000368499, 0.007357009133607575),
      (0.0017313803953611694, 0.0021865104888357995, 0.012039226730239348, 2.779898052155163E-4),
      (0.2955055316675138, 0.01762739164947898, 0.8542689284325417, 0.16897852768979024),
      (0.021470019698952052, 0.001960833716900792, 0.10093684687270144, 0.052205211671974075)
    )
    results._4 shouldBe DenseMatrix(
      (0.0, 0.0875428779442073, 0.0, 0.2814720521083398, 0.3832687232362885),
      (0.0, 7.649747594083709E-4, 0.0, 0.014708431655373989, 0.015470132660243439),
      (0.0, 0.8990820470485451, 0.0, 0.6389635896483882, 0.437298332390779),
      (0.0, 0.012610100247838865, 0.0, 0.06485592658789807, 0.16396281171268937)
    )
  }

  it should "produce the right results without 'ste' in params" in {
    val nComponents: Int = 4
    val nFeatures: Int = 5
    val startProb: Option[DenseVector[Double]] = Some(DenseVector(0.2, 0.05, 0.6, 0.15))
    val transMat: Option[DenseMatrix[Double]] = Some(DenseMatrix(
      (0.8, 0.1, 0.05, 0.05),
      (0.1, 0.7, 0.15, 0.05),
      (0.05, 0.1, 0.6, 0.25),
      (0.1, 0.05, 0.1, 0.75)
    ))
    val model = new MultinomialHMM(
      nComponents = nComponents,
      nFeatures = nFeatures,
      params = ""
    )
    val stats = new BaseHMM.Stats(
      0,
      DenseVector.zeros[Double](nComponents),
      DenseMatrix.zeros[Double](nComponents, nComponents),
      DenseMatrix.zeros[Double](nComponents, nFeatures)
    )
    val X = DenseVector[Int](1, 4, 3)
    val frameLogProb = log(DenseMatrix(
      (0.05, 0.01, 0.90, 0.04),
      (0.85, 0.02, 0.03, 0.10),
      (0.12, 0.03, 0.80, 0.05)
    ))
    val (loss, fwdLattice) = BaseHMM.doForwardPass(frameLogProb, startProb.get, transMat.get)
    val bwdLattice = BaseHMM.doBackwardPass(frameLogProb, startProb.get, transMat.get)
    val posteriors = BaseHMM.computePosteriors(fwdLattice, bwdLattice)
    val results = model.accumulateSufficientStatistics(stats, X, frameLogProb, posteriors, fwdLattice, bwdLattice)
    results._1 shouldBe 1
    results._2 shouldBe DenseVector.zeros[Double](nComponents)
    results._3 shouldBe DenseMatrix.zeros[Double](nComponents, nComponents)
    results._4 shouldBe DenseMatrix.zeros[Double](nComponents, nFeatures)
  }

  "scoreSamples" should "compute the right logProb and posteriors" in {
    val nComponents: Int = 4
    val nFeatures: Int = 5
    val startProb: Option[DenseVector[Double]] = Some(DenseVector(0.2, 0.05, 0.6, 0.15))
    val transMat: Option[DenseMatrix[Double]] = Some(DenseMatrix(
      (0.8, 0.1, 0.05, 0.05),
      (0.1, 0.7, 0.15, 0.05),
      (0.05, 0.1, 0.6, 0.25),
      (0.1, 0.05, 0.1, 0.75)
    ))
    val emissionProb: Option[DenseMatrix[Double]] = Some(DenseMatrix(
      (0.2, 0.2, 0.2, 0.2, 0.2),
      (0.1, 0.05, 0.3, 0.05, 0.5),
      (0.3, 0.3, 0.1, 0.1, 0.1),
      (0.2, 0.3, 0.1, 0.2, 0.2)
    ))
    val model = new MultinomialHMM(
      nComponents = nComponents,
      nFeatures = nFeatures,
      startProbPrior = startProb,
      transMatPrior = transMat,
      emissionProbPrior = emissionProb
    )
    val X = DenseVector(1, 0, 3, 3, 2, 1, 0)
    val lengths = DenseVector(4, 3)
    val (logProb, posteriors) = model.scoreSamples(X, Some(lengths))
    logProb shouldBe -11.537757324381989
    posteriors shouldBe DenseMatrix(
      (0.15503780311906423, 0.004216295917615701, 0.6560422976892534, 0.18470360327406654),
      (0.1944797184015473, 0.017634949905384653, 0.43567362835023016 , .35221170334283775),
      (0.2640155481027297, 0.018776587941909922, 0.19426734014916416, 0.5229405238061958),
      (0.31222457195394043, 0.030482784874890947, 0.13275697401283462, 0.5245356691583336),
      (0.2366118706648984, 0.05335216051444001, 0.5774856172421355, 0.1325503515785257),
      (0.2262314789195997, 0.023518075047942545, 0.47799414194804085, 0.27225630408441687),
      (0.23455105759781697, 0.04871533111350695, 0.40869930256742987, 0.30803430872124643)
    )
  }

}
