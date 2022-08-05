import scalismo.ui.api._

import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh._
import scalismo.io.{StatisticalModelIO, MeshIO}
import scalismo.statisticalmodel._
import scalismo.registration._
import scalismo.statisticalmodel.dataset._
import scalismo.numerics.PivotedCholesky.RelativeTolerance

object shapemodel extends App {

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  val dsGroup = ui.createGroup("files")

  val meshFiles = new java.io.File("files/3D_files/").listFiles
  val (meshes, meshViews) = meshFiles.map(meshFile => {
    val mesh = MeshIO.readMesh(meshFile).get
    val meshView = ui.show(dsGroup, mesh, "mesh")
    (mesh, meshView) // return a tuple of the mesh and the associated view
  }) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews

  val reference = meshes.head
  val toAlign : IndexedSeq[TriangleMesh[_3D]] = meshes.tail

  //val ptIds1 = Seq(PointId(37), PointId(1475), PointId(1099), PointId(2513), PointId(860), PointId(2314), PointId(368), PointId(1834),PointId(400), PointId(2009), PointId(5073), PointId(6146))

// landmark rigid alignement
  val pointIds = IndexedSeq(37, 1475, 1099, 2513, 860, 2314, 368, 1834, 400, 2009, 5073, 6146)
  val refLandmarks = pointIds.map{id => Landmark(s"L_$id", reference.pointSet.point(PointId(id))) }


  val alignedMeshes = toAlign.map { mesh =>
    val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
    val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks, refLandmarks, center = Point(0,0,0))
    mesh.transform(rigidTrans)
  }

  val meshView2 = ui.show(dsGroup, alignedMeshes(0), "head_02")
  val meshView3 = ui.show(dsGroup, alignedMeshes(1), "head_03")
  meshView2.color = java.awt.Color.RED

  meshView3.color = java.awt.Color.GREEN

// shape model
  val defFields = alignedMeshes.map{ m =>
    val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
      m.pointSet.point(id) - reference.pointSet.point(id)
    }.toIndexedSeq
    DiscreteField3D(reference, deformationVectors)
  }

  val continuousFields = defFields.map(f => f.interpolate(TriangleMeshInterpolator3D()) )
  val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference,
    continuousFields, RelativeTolerance(1e-8)
  )
  val model = PointDistributionModel(gp)
  val modelGroup = ui.createGroup("model")
  val ssmView = ui.show(modelGroup, model, "model")

/*
  val sampleGroup = ui.createGroup("samples")

  val meanFace : TriangleMesh[_3D] = model.mean
  ui.show(sampleGroup, meanFace, "meanFace")*/

  // moodel form data collection

  val dc = DataCollection.fromTriangleMesh3DSequence(reference, alignedMeshes)

  val modelFromDataCollection = PointDistributionModel.createUsingPCA(dc)

  val modelGroup2 = ui.createGroup("modelGroup2")
  ui.show(modelGroup2, modelFromDataCollection, "ModelDC")

  // General Procrustes analysis to align them better

  val dcWithGPAAlignedShapes = DataCollection.gpa(dc)
  val modelFromDataCollectionGPA = PointDistributionModel.createUsingPCA(dcWithGPAAlignedShapes)

  val modelGroup3 = ui.createGroup("modelGroup3")
  ui.show(modelGroup3, modelFromDataCollectionGPA, "ModelDCGPA")

}
