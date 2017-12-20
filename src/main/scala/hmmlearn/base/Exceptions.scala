package hmmlearn.base


case class ValueError(message: String) extends Exception(message, None.orNull)
