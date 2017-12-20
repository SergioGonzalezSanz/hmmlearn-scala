package hmmlearn.base

import org.scalatest.{FlatSpec, Matchers}

class ConvergenceMonitorTest extends FlatSpec with Matchers {

  "ConvergenceMonitor" should "converge by iterations" in {
    val monitor = new ConvergenceMonitor(tolerance = 1e-3, maxIterations = 2, verbose = false)
    monitor.converged shouldBe false
    monitor.report(-.01)
    monitor.converged shouldBe false
    monitor.report(-.1)
    monitor.converged shouldBe true
  }

  "ConvergenceMonitor" should "converge by logprob" in {
    val monitor = new ConvergenceMonitor(tolerance = 1e-3, maxIterations = 10, verbose = false)
    Array(-.03, -.02, -.01).foreach(logprob => {
      monitor.report(logprob)
      monitor.converged shouldBe false
    })
    monitor.report(-0.0101)
    monitor.converged shouldBe true
  }

}
