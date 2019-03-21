add_library(knn knn_handwriteDigitsClassifer.cpp)
target_link_libraries(knn ${3rdPart_LIBS})
add_executable(runTest runTest.cpp)
target_link_libraries(runTest knn) 