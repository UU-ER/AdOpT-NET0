.. _logging:

=========================
Logging
=========================

You can use the python logging module to log the AdOpT-NET0 during runtime.
Therefore, you can specify a logger as shown below:

.. testcode::

        import adopt_net0 as adopt
        import logging

        logger = logging.getLogger('adopt_net0')
        fh = logging.FileHandler("C:/Users/6574114/PycharmProjects/adopt_net0/log.txt")
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Specify the path to your input data
        path = "C:/Users/6574114/PycharmProjects/adopt_net0/test_cases/household_autarky"

        # Construct and solve the model
        m = adopt.ModelHub()
        m.read_data(path)
        m.quick_solve()
