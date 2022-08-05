import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation._
import scalismo.mesh._
import scalismo.registration._
import scalismo.io.MeshIO
import scalismo.numerics._
import scalismo.kernels._
import scalismo.statisticalmodel._
import breeze.linalg.DenseVector

import scalismo.geometry._
import scalismo.common._
import scalismo.mesh.TriangleMesh
import scalismo.transformations._
import scalismo.io.MeshIO
import scalismo.ui.api._

import scalismo.ui.api._

import breeze.linalg.{DenseVector}


object shapemodel extends App {

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  val referenceMesh = MeshIO.readMesh(new java.io.File("files/head_03.stl")).get
  val modelGroup = ui.createGroup("model")
  val refMeshView = ui.show(modelGroup, referenceMesh, "referenceMesh")
  refMeshView.color = java.awt.Color.GREEN

  val targetGroup = ui.createGroup("target")
  val targetMesh = MeshIO.readMesh(new java.io.File("files/head_01.stl")).get
  val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")
  targetMeshView.color = java.awt.Color.BLUE
  Thread.sleep(5000)

  // rigid alignment
  val ptIds1 = Seq(PointId(37), PointId(1475), PointId(1099), PointId(2513), PointId(860), PointId(2314), PointId(368), PointId(1834),PointId(400), PointId(2009), PointId(5073), PointId(6146))
  val head_01LM = ptIds1.map(pId => Landmark(s"lm-${pId.id}", referenceMesh.pointSet.point(pId)))
  //val head_01LM_view = head_01LM.map(lm => ui.show(modelGroup, lm, s"${lm.id}"))

  val ptIds2 = Seq(PointId(37), PointId(1475), PointId(1099), PointId(2513), PointId(860), PointId(2314), PointId(368), PointId(1834),PointId(400), PointId(2009), PointId(5073), PointId(6146))
  val head_03LM = ptIds2.map(pId => Landmark(s"lm-${pId.id}", targetMesh.pointSet.point(pId)))
  //val head_02LM_view = head_03LM.map(lm => ui.show(targetGroup, lm, s"${lm.id}"))

  import scalismo.registration.LandmarkRegistration
  val bestTransform1 : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(head_01LM, head_03LM, center = Point(0, 0, 0))
  //val bestTransform2 : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(head_03LM, head_01LM, center = Point(0, 0, 0))


  val transformedLms1 = head_01LM.map(lm => lm.transform(bestTransform1))
  //val landmarkViews1 = ui.show(modelGroup, transformedLms1, "transformLM_01")


  val alignedHead_02 = referenceMesh.transform(bestTransform1)
  val alignedHead_02_view = ui.show(modelGroup, alignedHead_02, "alignedHead_02")
  alignedHead_02_view.color = java.awt.Color.GREEN

  refMeshView.remove()
  targetMeshView.opacity = 0.5


  val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
  val kernel = DiagonalKernel3D(GaussianKernel3D(sigma = 70) * 50.0, outputDim = 3)
  val gp = GaussianProcess(zeroMean, kernel)

  val interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    alignedHead_02,
    gp,
    relativeTolerance = 0.05,
    interpolator = interpolator)

  val gpView = ui.addTransformation(modelGroup, lowRankGP, "gp")

/*
  val targetGroup = ui.createGroup("target")
  val targetMesh = MeshIO.readMesh(new java.io.File("files/head_02.stl")).get
  val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")*/

  val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)

  val fixedImage = alignedHead_02.operations.toDistanceImage
  val movingImage = targetMesh.operations.toDistanceImage
  val sampler = FixedPointsUniformMeshSampler3D(alignedHead_02, numberOfPoints = 1000)
  val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)

  val optimizer = LBFGSOptimizer(maxNumberOfIterations = 20)

  val regularizer = L2Regularizer(transformationSpace)

  val registration = Registration(metric, regularizer, regularizationWeight = 1e-5, optimizer)

  val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)
  val registrationIterator = registration.iterator(initialCoefficients)

  val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
    println(s"object value in iteration $itnum is ${it.value}")
    gpView.coefficients = it.parameters
    it
  }

  val registrationResult = visualizingRegistrationIterator.toSeq.last

  val registrationTransformation = transformationSpace.transformationForParameters(registrationResult.parameters)
  val fittedMesh = alignedHead_02.transform(registrationTransformation)

  val targetMeshOperations = targetMesh.operations
  val projection = (pt: Point[_3D]) => {
    targetMeshOperations.closestPointOnSurface(pt).point
  }

  val finalTransformation = registrationTransformation.andThen(projection)

  val projectedMesh = alignedHead_02.transform(finalTransformation)
  val resultGroup = ui.createGroup("result")
  val projectionView = ui.show(resultGroup, projectedMesh, "projection")

  case class RegistrationParameters(regularizationWeight: Double, numberOfIterations: Int, numberOfSampledPoints: Int)

  def doRegistration(
                      lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
                      referenceMesh: TriangleMesh[_3D],
                      targetmesh: TriangleMesh[_3D],
                      registrationParameters: RegistrationParameters,
                      initialCoefficients: DenseVector[Double]
                    ): DenseVector[Double] = {
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = FixedPointsUniformMeshSampler3D(
      referenceMesh,
      registrationParameters.numberOfSampledPoints
    )
    val metric = MeanSquaresMetric(
      fixedImage,
      movingImage,
      transformationSpace,
      sampler
    )
    val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
    val regularizer = L2Regularizer(transformationSpace)
    val registration = Registration(
      metric,
      regularizer,
      registrationParameters.regularizationWeight,
      optimizer
    )
    val registrationIterator = registration.iterator(initialCoefficients)
    val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
      println(s"object value in iteration $itnum is ${it.value}")
      it
    }
    val registrationResult = visualizingRegistrationIterator.toSeq.last
    registrationResult.parameters
  }
  val registrationParameters = Seq(
    RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 20, numberOfSampledPoints = 1000),
    RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 30, numberOfSampledPoints = 1000),
    RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 40, numberOfSampledPoints = 2000),
    RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50, numberOfSampledPoints = 4000)
  )


  val finalCoefficients = registrationParameters.foldLeft(initialCoefficients)((modelCoefficients, regParameters) =>
    doRegistration(lowRankGP, alignedHead_02, targetMesh, regParameters, modelCoefficients)
  )


}
