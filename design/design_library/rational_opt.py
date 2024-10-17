def rational_opt(ratio, error = 0.1, max_q = 10):
    # p, q -> p/q
    
    if ratio > 0:
        if ratio > 1:
            
            for q in range(1, max_q+1):
                p = round(q * ratio)
                
                if abs(p/q - ratio) < error:
                    return [p, q]
                
            else:
                raise RecursionError('Max Iterations Reached')
                
        elif ratio < 1:
            return rational_opt(1/ratio, error=error, max_q=max_q)[::-1]
        else:
            return [1, 1]
        
    else:
        raise NotImplementedError('Zero and Negative ratios are not supported')
    