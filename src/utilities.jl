
"""
    log_sum_exp(x)

The 'Bare Metal' stability trick for Logit/Softmax math.
"""
function log_sum_exp(x)
    max_x = maximum(x)
    s = 0.0
    for xi in x
        s += exp(xi - max_x)
    end
    return max_x + log(s)
end

"""
    log_mix(weights, f)

Zero-allocation log-mixture-density.
`f(j)` returns the log-density of component `j`.

Usage with `do` syntax:
    log_mix(theta) do j
        normal_lpdf(x[i], mus[j], sigma)
    end
"""
function log_mix(f, weights)
    K = length(weights)
    acc = log(weights[1]) + f(1)
    for j in 2:K
        lp_j = log(weights[j]) + f(j)
        if lp_j > acc
            acc = lp_j + log1p(exp(acc - lp_j))
        else
            acc = acc + log1p(exp(lp_j - acc))
        end
    end
    return acc
end