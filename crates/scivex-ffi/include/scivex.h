/*
 * scivex.h — C API for the Scivex scientific computing library.
 *
 * Link against -lscivex_ffi (dynamic) or libscivex_ffi.a (static).
 *
 * Memory: all opaque pointers returned by scivex_*_new / scivex_*_from_*
 * must be freed with the matching scivex_*_free function.
 *
 * Errors: functions returning int use 0 for success, -1 for error.
 * Call scivex_last_error() to get the error message.
 */

#ifndef SCIVEX_H
#define SCIVEX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Error handling
 * ====================================================================== */

/* Return a pointer to the last error message (thread-local), or NULL.
 * The string is valid until the next FFI call on the same thread.
 * The caller must NOT free this pointer. */
const char *scivex_last_error(void);

/* Clear the last error. */
void scivex_clear_error(void);

/* ======================================================================
 * Tensor (f64)
 * ====================================================================== */

/* Opaque handle to a Tensor<f64>. */
typedef struct ScivexTensor ScivexTensor;

/* Create a tensor from a flat data array and shape array.
 * Returns NULL on failure. */
ScivexTensor *scivex_tensor_from_array(const double *data, size_t data_len,
                                       const size_t *shape, size_t shape_len);

/* Create a tensor of zeros with the given shape. */
ScivexTensor *scivex_tensor_zeros(const size_t *shape, size_t shape_len);

/* Create a tensor of ones with the given shape. */
ScivexTensor *scivex_tensor_ones(const size_t *shape, size_t shape_len);

/* Create an identity matrix of size n x n. */
ScivexTensor *scivex_tensor_eye(size_t n);

/* Free a tensor. Accepts NULL safely. */
void scivex_tensor_free(ScivexTensor *t);

/* Return the number of dimensions. */
size_t scivex_tensor_ndim(const ScivexTensor *t);

/* Return the total number of elements. */
size_t scivex_tensor_numel(const ScivexTensor *t);

/* Write the shape into `out` (up to `out_len` values).
 * Returns the number of dimensions. */
size_t scivex_tensor_shape(const ScivexTensor *t, size_t *out, size_t out_len);

/* Copy tensor data into `out` (up to `out_len` elements). */
void scivex_tensor_data(const ScivexTensor *t, double *out, size_t out_len);

/* Return a pointer to the tensor's internal data buffer.
 * Valid until the tensor is freed or modified. */
const double *scivex_tensor_data_ptr(const ScivexTensor *t);

/* Element-wise addition. Returns NULL on shape mismatch. */
ScivexTensor *scivex_tensor_add(const ScivexTensor *a, const ScivexTensor *b);

/* Element-wise subtraction. Returns NULL on shape mismatch. */
ScivexTensor *scivex_tensor_sub(const ScivexTensor *a, const ScivexTensor *b);

/* Element-wise multiplication. Returns NULL on shape mismatch. */
ScivexTensor *scivex_tensor_mul(const ScivexTensor *a, const ScivexTensor *b);

/* Matrix multiplication. Returns NULL on dimension mismatch. */
ScivexTensor *scivex_tensor_matmul(const ScivexTensor *a, const ScivexTensor *b);

/* Dot product of two 1-D tensors. Returns 0 on success, -1 on error. */
int scivex_tensor_dot(const ScivexTensor *a, const ScivexTensor *b, double *out);

/* Matrix transpose. Returns NULL on error. */
ScivexTensor *scivex_tensor_transpose(const ScivexTensor *t);

/* Add a scalar to every element. */
ScivexTensor *scivex_tensor_add_scalar(const ScivexTensor *t, double s);

/* Multiply every element by a scalar. */
ScivexTensor *scivex_tensor_mul_scalar(const ScivexTensor *t, double s);

/* Sum of all elements. */
double scivex_tensor_sum(const ScivexTensor *t);

/* Mean of all elements. */
double scivex_tensor_mean(const ScivexTensor *t);

/* Determinant of a square matrix. Returns 0 on success, -1 on error. */
int scivex_tensor_det(const ScivexTensor *t, double *out);

/* Reshape a tensor. Returns NULL on invalid shape. */
ScivexTensor *scivex_tensor_reshape(const ScivexTensor *t,
                                    const size_t *shape, size_t shape_len);

/* Solve linear system A*x = b. Returns x or NULL on error. */
ScivexTensor *scivex_tensor_solve(const ScivexTensor *a, const ScivexTensor *b);

/* Matrix inverse. Returns NULL on error. */
ScivexTensor *scivex_tensor_inv(const ScivexTensor *t);

/* ======================================================================
 * Descriptive statistics (operate on raw arrays)
 * ====================================================================== */

/* Mean of data[0..len). Returns NaN on error. */
double scivex_stats_mean(const double *data, size_t len);

/* Sample variance (ddof=1). Returns NaN on error. */
double scivex_stats_variance(const double *data, size_t len);

/* Sample standard deviation (ddof=1). Returns NaN on error. */
double scivex_stats_std_dev(const double *data, size_t len);

/* Median. Returns NaN on error. */
double scivex_stats_median(const double *data, size_t len);

/* Pearson correlation. Returns 0 on success, -1 on error. */
int scivex_stats_pearson(const double *x, const double *y, size_t len,
                         double *out);

/* ======================================================================
 * Normal distribution
 * ====================================================================== */

typedef struct ScivexNormal ScivexNormal;

/* Create a Normal(mean, std_dev) distribution. Returns NULL if std_dev <= 0. */
ScivexNormal *scivex_normal_new(double mean, double std_dev);

/* Free a Normal distribution handle. Accepts NULL safely. */
void scivex_normal_free(ScivexNormal *n);

/* Probability density function at x. */
double scivex_normal_pdf(const ScivexNormal *n, double x);

/* Cumulative distribution function at x. */
double scivex_normal_cdf(const ScivexNormal *n, double x);

/* Inverse CDF (percent-point function) at probability p.
 * Returns 0 on success, -1 on error. */
int scivex_normal_ppf(const ScivexNormal *n, double p, double *out);

/* ======================================================================
 * Linear Regression
 * ====================================================================== */

typedef struct ScivexLinearRegression ScivexLinearRegression;

/* Create a new (unfitted) linear regression model. */
ScivexLinearRegression *scivex_linreg_new(void);

/* Free a linear regression model. Accepts NULL safely. */
void scivex_linreg_free(ScivexLinearRegression *m);

/* Fit the model. x: [n_samples, n_features], y: [n_samples].
 * Returns 0 on success, -1 on error. */
int scivex_linreg_fit(ScivexLinearRegression *m, const ScivexTensor *x,
                      const ScivexTensor *y);

/* Predict targets for x. Returns a new tensor or NULL on error. */
ScivexTensor *scivex_linreg_predict(const ScivexLinearRegression *m,
                                    const ScivexTensor *x);

/* Get fitted weights. Writes up to out_len values into out.
 * Returns the number of weights, or -1 if not fitted. */
int scivex_linreg_weights(const ScivexLinearRegression *m, double *out,
                          size_t out_len);

/* Get fitted bias (intercept). Returns 0 on success, -1 if not fitted. */
int scivex_linreg_bias(const ScivexLinearRegression *m, double *out);

/* ======================================================================
 * K-Means clustering
 * ====================================================================== */

typedef struct ScivexKMeans ScivexKMeans;

/* Create a KMeans model. Returns NULL on invalid parameters. */
ScivexKMeans *scivex_kmeans_new(size_t n_clusters, size_t max_iter,
                                size_t n_init, uint64_t seed);

/* Free a KMeans model. Accepts NULL safely. */
void scivex_kmeans_free(ScivexKMeans *m);

/* Fit the model. x: [n_samples, n_features].
 * Returns 0 on success, -1 on error. */
int scivex_kmeans_fit(ScivexKMeans *m, const ScivexTensor *x);

/* Predict cluster labels. Returns a new tensor or NULL on error. */
ScivexTensor *scivex_kmeans_predict(const ScivexKMeans *m,
                                    const ScivexTensor *x);

/* Get the inertia (sum of squared distances). Returns NaN if not fitted. */
double scivex_kmeans_inertia(const ScivexKMeans *m);

/* ======================================================================
 * Metrics
 * ====================================================================== */

/* Classification accuracy. Returns 0 on success, -1 on error. */
int scivex_metrics_accuracy(const double *y_true, const double *y_pred,
                            size_t len, double *out);

/* Mean squared error. Returns 0 on success, -1 on error. */
int scivex_metrics_mse(const double *y_true, const double *y_pred,
                       size_t len, double *out);

#ifdef __cplusplus
}
#endif

#endif /* SCIVEX_H */
