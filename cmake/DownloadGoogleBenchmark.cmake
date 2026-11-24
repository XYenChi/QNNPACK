# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

# Benchmark 1.9.3 adds RISC-V cycle-counter support and requires C++17.

PROJECT(googlebenchmark-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googlebenchmark
	URL https://github.com/google/benchmark/archive/refs/tags/v1.9.3.zip
	URL_HASH SHA256=ba944d96e4d3ecaa84487dc26a81c43db1728734cf80123e847a370f7e4bf281
	SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark"
	BINARY_DIR "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
