"""Convert structured reaction-step JSON into reaction skeletons."""

from __future__ import annotations


def expand_reactants(current_reactants: set[str], intermediates: dict[str, set[str]]) -> set[str]:
    expanded = set()
    for reactant in current_reactants:
        if reactant in intermediates:
            expanded.update(expand_reactants(intermediates[reactant], intermediates))
        else:
            expanded.add(reactant)
    return expanded


def make_intermediates_list(response: dict):
    if not isinstance(response, dict):
        raise TypeError("response should be a dictionary.")

    product_key = "Product" if "Product" in response else "Products"
    products_list = list(response[product_key].keys())
    product_reactants = {}
    intermediates = {}
    reaction_steps = set()
    work_up_involve_chemicals = set()
    work_up_chemicals = set()
    reaction_label = []
    work_up_label = []
    work_up_involve_label = []

    for step, equation in response["Reaction Steps"].items():
        reactants_raw, products_raw = equation.split("->")
        reactants = set(reactants_raw.split("+"))
        products = products_raw.split("+")

        step_index = int(step.split(" ")[0]) - 1
        if "reaction" in step.lower():
            reaction_label.append(step_index)
            reaction_steps.update(reactants, products)
        elif "work-up-involve" in step.lower():
            work_up_involve_label.append(step_index)
            work_up_involve_chemicals.update(reactants, products)
        else:
            work_up_label.append(step_index)
            work_up_chemicals.update(reactants, products)

        for product in products:
            if product in response[product_key]:
                product_reactants[product] = expand_reactants(reactants, intermediates)
            intermediates[product] = reactants

    grouped = {}
    for _index, name, reactant_set in zip(
        range(len(intermediates)),
        intermediates.keys(),
        intermediates.values(),
    ):
        grouped.setdefault(frozenset(reactant_set), []).append(name)

    intermediates_list = []
    for idx, (reactant_set, names) in enumerate(grouped.items()):
        intermediates_list.append((idx, ".".join(sorted(names)), reactant_set))

    work_up_chemicals = work_up_chemicals - set(products_list) - set(reaction_steps)
    return (
        intermediates_list,
        products_list,
        work_up_chemicals,
        work_up_label,
        work_up_involve_chemicals,
        work_up_involve_label,
        reaction_label,
    )


def filter_intermediates_list(intermediates_list: list[tuple[int, str, frozenset[str]]]):
    filtered_intermediates = []
    for step, intermediates, reactants in intermediates_list:
        filtered_reactants = frozenset(
            reactant for reactant in reactants if "mixture" not in reactant
        )
        filtered_intermediates.append((step, intermediates, filtered_reactants))
    return filtered_intermediates


def segmentation(intermediates_list, products_list):
    breakpoints = set()
    for idx, intermediates in enumerate(intermediates_list):
        if any(product in intermediates[1] for product in products_list):
            breakpoints.add(idx)

    segments = []
    start = 0
    for point in sorted(breakpoints):
        segments.append(list(range(start, point + 1)))
        start = point + 1
    return segments


def join_reaction(data) -> str:
    result = []
    for item in data:
        if isinstance(item, frozenset):
            filtered = [value for value in sorted(item) if "mixture" not in value]
            result.append(".".join(filtered))
        elif isinstance(item, str):
            result.append(item)
    return "".join(result)


def process_reaction_data(data: dict) -> list[str] | None:
    try:
        (
            intermediates_list,
            products_list,
            _work_up_chemicals,
            _work_up_label,
            _work_up_involve_chemicals,
            work_up_involve_label,
            reaction_label,
        ) = make_intermediates_list(data)

        final_result = {}
        filtered_intermediates = filter_intermediates_list(intermediates_list)

        for steps_list in segmentation(intermediates_list, products_list):
            temp_reaction_equation = []
            product = intermediates_list[steps_list[-1]][1]
            for step in steps_list:
                reactants = filtered_intermediates[step][2]
                if step in reaction_label and reactants:
                    temp_reaction_equation.extend([reactants, ">"])
                elif step in work_up_involve_label and reactants:
                    temp_reaction_equation.extend([">", reactants, ">"])
            final_result[product] = join_reaction(temp_reaction_equation)

        final_reaction_formulas = {}
        for product, rxn_eqn in final_result.items():
            rxn_eqn = rxn_eqn.lstrip(">")
            final_reaction_formulas[product] = f"{rxn_eqn}>{product}".replace(">>", ">")

        return [
            rxn_eqn
            for rxn_eqn in final_reaction_formulas.values()
            if not rxn_eqn.startswith(">")
        ]
    except Exception as exc:
        print(
            "Error: "
            + str(exc)
            + ";reactant from the previous step is stated as the product in the next step"
        )
        return None
