function extract_result(string)
    regex = r"\d+(?:\.\d+)?(?:[eE][+-]\d+)?"
    numbers = map(eachmatch(regex, string)) do m
        parse(Float64, m.match)
    end
    if length(numbers) <= 5
        return [(numbers[1], numbers[2]), numbers[3:end]...]
    else
        return [(numbers[1], numbers[2], numbers[3]), numbers[4:end]...]
    end
end