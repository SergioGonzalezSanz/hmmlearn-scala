package hmmlearn.base

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, Matchers}


class OptimisationTest extends FlatSpec with Matchers {

  "forward" should "compute the right results" in {
    val nSamples = 10
    val nComponents = 4
    val logStartProb = DenseVector(-1.897, -1.050, -1.386, -1.386)
    val logTransMat = DenseMatrix(
      (-1.897, -1.050, -1.386, -1.386),
      (-2.303, -1.204, -1.609, -0.511),
      (-2.303, -1.204, -1.609, -0.511),
      (-1.897, -1.050, -1.386, -1.386)
    )
    val frameLogProb = DenseMatrix(
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895),
      (-1.504, -1.332, -2.098, -1.895)
    )
    val fwdLattice = Optimisation.forward(nSamples, nComponents, logStartProb, logTransMat, frameLogProb)
    fwdLattice shouldBe DenseMatrix(
      (-3.401, -2.382, -3.484, -3.2809999999999997),
      (-5.278839807540958, -4.116899331713416, -5.259582441613247, -4.285814946810763),
      (-6.806766634010698, -5.6711478773995205, -6.8066497964585935, -5.934300438982435),
      (-8.394003957984927, -7.254502648263259, -8.391072474215854, -7.503022413367903),
      (-9.972368404738807, -8.833461697568376, -9.969868171331854, -9.08420493431029),
      (-11.552088842027745, -10.413091583544423, -11.54952293856513, -10.663495805753206),
      (-13.13160271047798, -11.992619253796786, -13.129046816529772, -12.243075139086281),
      (-14.71114806253652, -13.572162502464334, -14.708590643143417, -13.822610514095556),
      (-16.29068861646553, -15.151703376955968, -16.288131429554518, -15.402152588549487),
      (-17.87022990164196,-16.731244613277912, -17.867672679300142, -16.981693641994287)
    )
  }

  "computeLogXiSum" should "provide the right results" in {
    val logXiSum = Optimisation.computeLogXiSum(
      3,
      4,
      DenseMatrix(
        (-4.605170185988091, -7.600902459542082, -0.6161861394238171, -5.115995809754082),
        (-3.496525061619607, -6.800696206688729, -4.6299496773523146, -4.268519393880286),
        (-5.760683833302373, -8.712296082574925, -4.940869521374431, -7.2285486981624025)
      ),
      DenseMatrix(
        (-0.2231435513142097, -2.3025850929940455, -2.995732273553991, -2.995732273553991),
        (-2.3025850929940455, -0.35667494393873245, -1.8971199848858813, -2.995732273553991),
        (-2.995732273553991, -2.3025850929940455, -0.5108256237659907, -1.3862943611198906),
        (-2.3025850929940455, -2.995732273553991, -2.3025850929940455, -0.2876820724517809)
      ),
      DenseMatrix(
        (-2.3234181009501014, -4.06772697431783, -3.9831565593989087, -3.7502230843549875),
        (-1.9554555618988445, -1.8611095473628483, -0.6901516715801467, -2.032557955780985),
        (0.0, 0.0, 0.0,  0.0)
    ),
      DenseMatrix(
        (-2.995732273553991, -4.605170185988091, -0.10536051565782628, -3.2188758248682006),
        (-0.16251892949777494, -3.912023005428146, -3.506557897319982, -2.3025850929940455),
        (-2.120263536200091, -3.506557897319982, -0.2231435513142097, -2.995732273553991)
      )
    )
    logXiSum shouldBe DenseMatrix(
      (0.297162374727239, 0.00836871289364116, 0.10347396524344381, 0.0073300783479383355),
      (0.0017298832841213872, 0.0021841235535164124, 0.011967331705208344, 2.7795117320898456E-4),
      (0.2589009909202237, 0.01747383113597579, 0.6174905097164667, 0.1561303142201153),
      (0.021242785553555504, 0.0019589137918276617, 0.09616149630676196, 0.05088816342145737)
    )
  }

  "logAddExp" should "compute the right results" in {
    Optimisation.logAddExp(0.5, 0.1) shouldBe 1.0130152523999527
    Optimisation.logAddExp(0.1, 0.5) shouldBe 1.0130152523999527
    Optimisation.logAddExp(-0.1, 0.5) shouldBe 0.9374879504858856
    Optimisation.logAddExp(-0.5, 0.5) shouldBe 0.8132616875182228
    Optimisation.logAddExp(Double.NegativeInfinity, 0.1) shouldBe 0.1
    Optimisation.logAddExp(0.5, Double.NegativeInfinity) shouldBe 0.5
  }

  "viterbi" should "produce the right results" in {
    val logStartProb = DenseVector(-0.69314718, -0.69314718)
    val logTransMat = DenseMatrix(
      (-0.35667494, -1.2039728),
      (-1.2039728, -0.35667494)
    )
    val frameLogProb = DenseMatrix(
      (-0.10536052, -1.6094379),
      (-0.10536052, -1.6094379),
      (-2.30258509, -0.22314355),
      (-0.10536052, -1.6094379),
      (-0.10536052, -1.6094379)
    )

    val (expectedLogProb, expectedStateSequence) = (-4.459028291034797, DenseVector(0, 0, 1, 0, 0))

    val (actualStateSequence, actualLogProb) = Optimisation.viterbi(
      nSamples = frameLogProb.rows,
      nComponents = frameLogProb.cols,
      logStartProb = logStartProb,
      logTransMat = logTransMat,
      frameLogProb = frameLogProb
    )

    round(actualLogProb) shouldBe round(expectedLogProb)
    actualStateSequence shouldBe expectedStateSequence
  }

  def round(d : Double) : Double = {
    (d * 10000).toInt / 10000.0
  }
}
