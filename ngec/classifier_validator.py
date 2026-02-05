"""
Validator for custom event classifiers.

Run this to check your classifier before using it in the pipeline.
"""


def validate_event_classifier(classifier, verbose=True):
    """
    Validate that a classifier has the required interface.

    This function checks that a classifier implements the expected interface
    and produces outputs in the correct format. Run this before using a custom
    classifier in the PloverCoder pipeline.

    Parameters
    ----------
    classifier : object
        The classifier instance to validate.
    verbose : bool
        If True, print detailed results (default: True).

    Returns
    -------
    list of str
        List of error messages (empty if valid).

    Example
    -------
    >>> from my_classifier import MyEventClassifier
    >>> clf = MyEventClassifier()
    >>> errors = validate_event_classifier(clf)
    >>> if not errors:
    ...     print("Classifier is valid!")

    >>> # Use in PloverCoder
    >>> from ngec import PloverCoder, validate_event_classifier
    >>> clf = MyEventClassifier()
    >>> errors = validate_event_classifier(clf)
    >>> if not errors:
    ...     coder = PloverCoder(es_client, event_classifier=clf)
    """
    errors = []
    warnings_list = []

    # Check 1: Has process() method
    if not hasattr(classifier, 'process'):
        errors.append("Missing required method: process(story_list)")
    elif not callable(getattr(classifier, 'process')):
        errors.append("'process' exists but is not callable")

    # Check 2: Test process() with sample data
    if hasattr(classifier, 'process') and callable(classifier.process):
        test_stories = [
            {'event_text': 'Protesters marched through the streets.', 'id': 'test1'},
            {'event_text': 'The president signed the agreement.', 'id': 'test2'},
        ]
        try:
            results = classifier.process(test_stories)

            # Check return type
            if not isinstance(results, list):
                errors.append(f"process() should return a list, got {type(results).__name__}")
            elif len(results) != 2:
                errors.append(f"process() returned {len(results)} items, expected 2")
            else:
                # Check each result has required keys
                for i, result in enumerate(results):
                    if not isinstance(result, dict):
                        errors.append(f"Result {i} should be a dict, got {type(result).__name__}")
                        continue

                    if 'event_type' not in result:
                        errors.append(f"Result {i} missing 'event_type' key")
                    elif not isinstance(result['event_type'], list):
                        errors.append(f"Result {i} 'event_type' should be a list, got {type(result['event_type']).__name__}")
                    else:
                        # Check event_type values are strings
                        for et in result['event_type']:
                            if not isinstance(et, str):
                                errors.append(f"Result {i} 'event_type' contains non-string: {type(et).__name__}")
                                break

                    if 'event_type_confidence' not in result:
                        warnings_list.append(f"Result {i} missing 'event_type_confidence' (optional but recommended)")
                    elif not isinstance(result['event_type_confidence'], dict):
                        warnings_list.append(f"Result {i} 'event_type_confidence' should be a dict")

                    # event_mode is optional (not all ontologies have modes)
                    if 'event_mode' in result and not isinstance(result['event_mode'], list):
                        errors.append(f"Result {i} 'event_mode' should be a list if present")

        except Exception as e:
            errors.append(f"process() raised an exception: {e}")

    # Check 3: Optional classify_one() method
    if hasattr(classifier, 'classify_one'):
        if not callable(getattr(classifier, 'classify_one')):
            warnings_list.append("'classify_one' exists but is not callable")
        else:
            try:
                result = classifier.classify_one("Test text for validation")
                if not isinstance(result, tuple):
                    warnings_list.append(f"classify_one() should return a tuple, got {type(result).__name__}")
                elif len(result) != 2:
                    warnings_list.append(f"classify_one() should return (types, modes) tuple with 2 elements")
                else:
                    types, modes = result
                    if not isinstance(types, list):
                        warnings_list.append(f"classify_one() first return value should be a list")
                    if not isinstance(modes, list):
                        warnings_list.append(f"classify_one() second return value should be a list")
            except Exception as e:
                warnings_list.append(f"classify_one() exists but raised: {e}")

    # Print results
    if verbose:
        if errors:
            print("Validation FAILED:")
            for err in errors:
                print(f"   ERROR: {err}")
        if warnings_list:
            print("Warnings:")
            for warn in warnings_list:
                print(f"   WARNING: {warn}")
        if not errors and not warnings_list:
            print("Classifier is valid!")
        elif not errors:
            print("Classifier is valid (with warnings)")

    return errors
