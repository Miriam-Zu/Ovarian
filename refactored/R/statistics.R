run_t_test <- function(values_h, values_oc, test_name) {

  p_var <- var.test(values_h, values_oc)$p.value
  var_equal <- p_var >= 0.05

  t_res <- t.test(values_h, values_oc, var.equal = var_equal)

  data.frame(
    test_name = test_name,
    mean_h = mean(values_h),
    mean_oc = mean(values_oc),
    var_test_p = p_var,
    var_equal = var_equal,
    t_statistic = t_res$statistic,
    p_value = t_res$p.value,
    conf_low = t_res$conf.int[1],
    conf_high = t_res$conf.int[2],
    stringsAsFactors = FALSE
  )
}