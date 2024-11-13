/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.kotlin.ml.classification

import android.app.Activity
import android.graphics.Bitmap
import android.media.Image
import com.google.android.gms.tasks.TaskExecutors
import com.google.ar.core.examples.kotlin.ml.classification.utils.ImageUtils
import com.google.ar.core.examples.kotlin.ml.third_party.YuvToRgbConverter
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import kotlinx.coroutines.tasks.await

/** Analyzes an image using ML Kit. */
class MLKitSegmenter(context: Activity) {
  // To use a custom model, follow steps on
  // https://developers.google.com/ml-kit/vision/object-detection/custom-models/android.
  // val model = LocalModel.Builder().setAssetFilePath("inception_v4_1_metadata_1.tflite").build()
  // val builder = CustomObjectDetectorOptions.Builder(model)

  // For the ML Kit default model, use the following:
  val builder = SelfieSegmenterOptions.Builder()

  private val options =
    builder
      .setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
      .build()
  private val detector = Segmentation.getClient(options)

  suspend fun analyze(image: Image, imageRotation: Int): SegmentationMask {
    // `image` is in YUV
    // (https://developers.google.com/ar/reference/java/com/google/ar/core/Frame#acquireCameraImage()),
    val convertYuv = convertYuv(image)

    // The model performs best on upright images, so rotate it.
    val rotatedImage = ImageUtils.rotateBitmap(convertYuv, imageRotation)

    val inputImage = InputImage.fromBitmap(rotatedImage, 0)

    return detector.process(inputImage).await()
  }

  val yuvConverter = YuvToRgbConverter(context)

  fun convertYuv(image: Image): Bitmap {
    return Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888).apply {
      yuvConverter.yuvToRgb(image, this)
    }
  }

  @Suppress("USELESS_IS_CHECK")
  fun hasCustomModel() = builder is CustomObjectDetectorOptions.Builder
}
