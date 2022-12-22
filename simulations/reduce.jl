using Printf

function reduce(n, every)
    for i = 0:n-1
        if !(i % every == 0)
            rm(@sprintf("%06d.png", i+1))
        end
    end
end

function rename(n, every)
    for i = 0:every:n-1
        mv(@sprintf("%06d.png", i+1), "frame-$(round(Int, i / every)+1).png"; force=true)
    end
end

# reduce(45, 1)
rename(45, 1)