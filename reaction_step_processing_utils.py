def expand_reactants(current_reactants, intermediates):
    expanded = set()
    for reactant in current_reactants:
        if reactant in intermediates:
            # Recursively expand reactant if it's an intermediate
            expanded.update(expand_reactants(intermediates[reactant], intermediates))
        else:
            expanded.add(reactant)
    return expanded

def make_intermediates_list(response):
    product_reactants = {}
    products_list = list(response["Product"].keys())
    intermediates = {}

    reaction_steps = set()
    work_up_invovle_chemicals = set()
    work_up_chemicals = set()

    reaction_label =[]
    work_up_label = []
    work_up_invovle_label = []

    if not isinstance(response, dict):
        raise TypeError("response should be a dictionary.")
    if not isinstance(products_list, list):
        raise TypeError("products_list should be a list.")
    
    # Process each reaction step
    for step, equation in response["Reaction Steps"].items():
        reactants, products = equation.split("->")
        reactants = set(reactants.split("+"))
        products = products.split("+")    
        if "Reaction" in step or "reaction" in step:
            reaction_label.append(int(step.split(' ')[0])-1)
            reaction_steps.update(reactants, products)
        elif "Work-up-involve" in step or "work-up-involve" in step:
            work_up_invovle_label.append(int(step.split(' ')[0])-1)
            work_up_invovle_chemicals.update(reactants, products)
        else:
            work_up_label.append(int(step.split(' ')[0])-1)
            work_up_chemicals.update(reactants, products)
        # Store intermediate products
        for product in products:
            if product in response["Product"]:
                # If the product is a final product, record the expanded reactants
                expanded_reactants = expand_reactants(reactants, intermediates)
                product_reactants[product] = expanded_reactants
            # Update or add new intermediates including current reactants
            intermediates[product] = reactants

    temp_intermediates_list = list(zip(range(len(intermediates)),intermediates.keys(),intermediates.values()))
    grouped = {}
    for index, name, sets in temp_intermediates_list:
        # Convert set to a frozenset for use as a dictionary key
        key = frozenset(sets)
        if key in grouped:
            grouped[key].append(name)
        else:
            grouped[key] = [name]

    intermediates_list = []
    for i, (sets, names) in enumerate(grouped.items()):
    # Join names with a dot if there are multiple names for the same set
        merged_name = '.'.join(sorted(names))
        intermediates_list.append((i, merged_name, sets))

    work_up_chemicals = work_up_chemicals - set(products_list)
    work_up_chemicals = work_up_chemicals - set(reaction_steps)
    return intermediates_list, products_list, work_up_chemicals, work_up_label, work_up_invovle_chemicals, work_up_invovle_label, reaction_label

def filter_intermediates_list(intermediates_list):
    # Create a new list to store updated data
    new_intermediates_list = []
    
    for step, intermediates, reactants in intermediates_list:
        # Create a new set for filtered reactants
        new_reactants_set = set()
        
        # Filter out reactants containing "mixture"
        for reactant in reactants:
            if 'mixture' not in reactant:
                new_reactants_set.add(reactant)

        # Append the updated tuple to the new list
        new_intermediates_list.append((step, intermediates, frozenset(new_reactants_set)))
    
    return new_intermediates_list

def segmentation(intermediates_list,products_list):
    breakpoints = set()
    for i, intermediates in enumerate(intermediates_list):
        if any(product in intermediates[1] for product in products_list):
            breakpoints.add(i)
                
    segments = []
    start = 0
    for point in sorted(breakpoints):
        segments.append(list(range(start,point+1)))
        start = point+1
    return segments

def join_reaction(data):
    result = []
    for item in data:
        if isinstance(item, frozenset):
            # Convert set to sorted list and then to string
            sorted_items = sorted(item)
            filtered = []
            for temp in sorted_items:
                if 'mixture' not in temp:
                    filtered.append(temp)
            joined_string = '.'.join(filtered)
            result.append(joined_string)
        elif isinstance(item, str):
            # Directly append strings and symbols
            result.append(item)
    return ''.join(result)
        
        
def process_reaction_data(data):
    try:
        intermediates_list, products_list, work_up_chemicals, work_up_label, work_up_invovle_chemicals, work_up_involve_label, reaction_label = make_intermediates_list(data)
        
        final_result = {}
    
        for steps_list in segmentation(intermediates_list, products_list):
            temp_reaction_equation = []
            product = intermediates_list[steps_list[-1]][1]
            final_result[product] = temp_reaction_equation
            for step in steps_list:
                if step in reaction_label and len(filter_intermediates_list(intermediates_list)[step][2]) != 0:
                    temp_reaction_equation.append(filter_intermediates_list(intermediates_list)[step][2])
                    temp_reaction_equation.append('>')
                elif step in work_up_involve_label and len(filter_intermediates_list(intermediates_list)[step][2]) != 0:
                        temp_reaction_equation.append('>')
                        temp_reaction_equation.append(filter_intermediates_list(intermediates_list)[step][2])
                        temp_reaction_equation.append('>')            
            final_result[product] = join_reaction(temp_reaction_equation)
        final_reaction_formulas = {}
        for product, rxn_eqn in final_result.items():
            if rxn_eqn.startswith('>'):
                rxn_eqn = rxn_eqn.lstrip('>')
            if rxn_eqn.endswith('>'):
                final_reaction_formulas[product] = (rxn_eqn+ product).replace('>>','>')
            if rxn_eqn.endswith(''):
                final_reaction_formulas[product] = (rxn_eqn+ '>'+ product).replace('>>','>')
        
        for product, rxn_eqn in list(final_reaction_formulas.items()):
             if rxn_eqn.startswith('>'):
                 del final_reaction_formulas[product]   
        return list(final_reaction_formulas.values())
    
    except Exception as e:
            error_message = "Error: "+ str(e) +';reactant from the previous step is stated as the product in the next step'
            print(error_message)
            return None