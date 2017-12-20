package hmmlearn.base


class ConvergenceMonitor(
                          tolerance: Double,
                          maxIterations: Int,
                          verbose: Boolean
                        ) {

  var previousLobprob: Option[Double] = None
  var currentLobprob: Option[Double] = None
  var iterations: Int = 0


  def report(logprob: Double): String = {
    val delta = if (previousLobprob.isDefined) previousLobprob.get - currentLobprob.get else Double.NaN
    previousLobprob = currentLobprob
    currentLobprob = Some(logprob)
    iterations += 1
    f"$iterations%10d $logprob%16.4f $delta%16.4f"
  }

  def converged: Boolean = {
    (iterations >= maxIterations) || ((previousLobprob.isDefined) && ((currentLobprob.get - previousLobprob.get) < tolerance))
  }
}
