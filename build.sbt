enablePlugins(BintrayPlugin)

name := "hmmlearn-scala"

version := "0.0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.slf4j" % "slf4j-api" % "1.7.25",
  "org.slf4j" % "slf4j-simple" % "1.7.25",
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scoverage" %% "scalac-scoverage-runtime" % "1.3.1"
)

val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % "3.0.1",
  "org.scalacheck" %% "scalacheck" % "1.12.2"
)

licenses += ("MIT", url("http://opensource.org/licenses/MIT"))
bintrayRepository := "hmm"
libraryDependencies ++= testDependencies

val publishRelease = TaskKey[Unit]("publishRelease", "Publish release to Nexus if tagged")
publishRelease := Def.taskDyn {
  val v = version.value
  val s = streams.value
  if (v.trim.endsWith("SNAPSHOT")) {
    Def.task(s.log.error(s"Can't publish SNAPSHOT version: $v"))
  }
  else {
    publish.toTask
  }
}.value
publishRelease <<= publishRelease.dependsOn(test in Test)
publishMavenStyle := true
bintrayPackage := name.value

// test command will now include running scalastyle for test files
lazy val testScalastyle = taskKey[Unit]("testScalastyle")
testScalastyle := scalastyle.in(Test).toTask("").value
(test in Test) := ((test in Test) dependsOn testScalastyle).value

// compile command will now include running scalastyle for src files
lazy val compileScalastyle = taskKey[Unit]("compileScalastyle")
compileScalastyle := scalastyle.in(Compile).toTask("").value
(compile in Compile) := ((compile in Compile) dependsOn compileScalastyle).value
