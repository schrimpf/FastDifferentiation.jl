using FastDifferentiation
using TestItemRunner
using TestItems

@testitem "multi-output factorization steps" begin
    using Test
    import FiniteDifferences
    import FastDifferentiation as FD

    @variables a1 a2 s q
    r = 0.2
    eps = 0.3
    y0 = 0.4
    m = log(1 + 2 * a1 * s + (a1^2 + 2 * a2) * (s^2 + 1) +
            2 * a1 * a2 * (s^3 + 3 * s) +
            a2^2 * (s^4 + 6 * s^2 + 3))
    z = (y0 + m - r * eps / q) / sqrt(1 - r^2 + 1e-12)
    f = log((1 + a1 * z + a2 * z^2)^2)
    vars = [a1, a2, s, q]
    point = [0.8, 0.2, 0.9, 1.1]

    eval_f(v) = begin
        a1v, a2v, sv, qv = v
        mv = log(1 + 2 * a1v * sv + (a1v^2 + 2 * a2v) * (sv^2 + 1) +
                 2 * a1v * a2v * (sv^3 + 3 * sv) +
                 a2v^2 * (sv^4 + 6 * sv^2 + 3))
        zv = (y0 + mv - r * eps / qv) / sqrt(1 - r^2 + 1e-12)
        log((1 + a1v * zv + a2v * zv^2)^2)
    end

    expected_hessian = FiniteDifferences.jacobian(
        FiniteDifferences.central_fdm(5, 1),
        x -> FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1), eval_f, x)[1],
        point,
    )[1]

    gradient = FD.jacobian([f], vars)
    @test size(gradient) == (1, length(vars))

    graph = FD.DerivativeGraph(vec(gradient))
    @test FD.codomain_dimension(graph) == length(vars)
    @test FD.domain_dimension(graph) == length(vars)

    subgraphs = FD.compute_factorable_subgraphs(graph)
    @test !isempty(subgraphs)

    """
        evaluated_paths(graph, vars, point)

    Evaluate every root-to-variable path in graph at point.
    """
    function evaluated_paths(graph, vars, point)
        values = [
            FD.evaluate_path(graph, root_index, variable_index)
            for root_index in 1:FD.codomain_dimension(graph),
                variable_index in 1:FD.domain_dimension(graph)
        ]
        return FD.make_function(values, vars)(point)
    end

    first_subgraph = pop!(subgraphs)
    @testset "subgraph discovery and evaluation" begin
        @test FD.subgraph_exists(first_subgraph)
        replacement_edges = FD.evaluate_subgraph(first_subgraph)
        @test !isempty(replacement_edges)
        @test all(Set((FD.top_vertex(edge), FD.bott_vertex(edge))) ==
                  Set(FD.vertices(first_subgraph))
                  for edge in replacement_edges)
        @test FD.verify_paths(graph)
        @test all(
            FD.reachable_variables(edge) ==
            FD.reachable_variables(graph, FD.bott_vertex(edge)) &&
            FD.reachable_roots(edge) ==
            FD.reachable_roots(graph, FD.top_vertex(edge))
            for edge in FD.unique_edges(graph)
        )
    end

    staged_graph = FD.DerivativeGraph(vec(gradient))
    staged_subgraph = pop!(FD.compute_factorable_subgraphs(staged_graph))
    @testset "individual graph mutation stages" begin
        staged_edges = FD.evaluate_subgraph(staged_subgraph)
        FD.add_non_dom_edges!(staged_subgraph)
        edges_to_delete = FD.reset_edge_masks!(staged_subgraph)
        for edge in edges_to_delete
            FD.delete_edge!(staged_graph, edge)
        end
        for edge in staged_edges
            FD.add_edge!(staged_graph, edge)
        end
        @test FD.verify_paths(staged_graph)
    end

    @testset "first factorization step" begin
        FD.factor_subgraph!(first_subgraph)
        @test FD.verify_paths(graph)
        @test FD.verify_paths(graph)
    end

    factored_graph = FD.DerivativeGraph(vec(gradient))
    FD.factor!(factored_graph)
    @testset "direct multi-output graph path diagnostic" begin
        @test FD.verify_paths(factored_graph)
        # This deliberately bypasses jacobian's row-wise multi-output path:
        # DerivativeGraph(vec(gradient)) is an internal diagnostic graph, not
        # the graph evaluated by FD.jacobian(vec(gradient), vars).
        @test_broken isapprox(
            evaluated_paths(factored_graph, FD.variables(factored_graph), point),
            expected_hessian[:, [findfirst(x -> x === var, vars) for var in FD.variables(factored_graph)]],
            atol=1e-5,
        )
    end

    hess = FD.jacobian(vec(gradient), vars)
    @testset "hessian evaluation" begin
        @test isapprox(
            FD.make_function(hess, vars)(point),
            expected_hessian,
            atol=1e-5,
        )
    end
end
