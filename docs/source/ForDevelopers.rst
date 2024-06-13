.. _developers-guide:

Developer Instruction
=====================================
This page contains general instructions for the developers that are working on the PyHub.

Coding conventions
-----------------------
To keep the code consistent and clear for other developers, try to use the coding conventions that are described in this \
section as much as possible.

For the Pyomo classes we use:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Objective   | objective... |
+-------------+--------------+
| Constraint  | const...     |
+-------------+--------------+
| Piecewise   | const...     |
+-------------+--------------+
| Set         | set...       |
+-------------+--------------+
| Block       | b...         |
+-------------+--------------+
| Var         | var...       |
+-------------+--------------+
| Param       | para...      |
+-------------+--------------+
| Disjunct    | dis...       |
+-------------+--------------+
| Disjunction | disjunction..|
+-------------+--------------+
| rule        | init...      |
+-------------+--------------+
| unit        | u            |
+-------------+--------------+

Other names that are regularly used in AdOpT-NET0 are:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Timestep    | t...         |
+-------------+--------------+
| Carrier     | car...       |
+-------------+--------------+
| Node        | node...      |
+-------------+--------------+
| Network     | netw...      |
+-------------+--------------+
| Technology  | tec...       |
+-------------+--------------+
| Consumption | cons...      |
+-------------+--------------+
| Input       | input...     |
+-------------+--------------+
| Output      | output...    |
+-------------+--------------+

Document your code!
-------------------
Please make sure to document your code properly. Each function should have a docstring at the beginning \
that shortly describes what the function does as well as input and return variables. This docstring \
is meant to appear in this documentation and should be written in a way that can be understood by \
users and not only developers. In addition, include comments in your code that are valuable hints for \
people reading your code. To create a new version of this website, you need to have \
`sphinx <https://sphinx-tutorial.readthedocs.io/>`_, a documentation tool for python. To create an \
html documentation website, you need to move to the ``.\docs`` folder in your terminal and execute \
either `.\make html` or simply `make html`.
We refer to the following guides on documentation:

* `PEP 8 - Style Guide for Python Code <https://peps.python.org/pep-0008/>`_
* `PEP 257 <https://peps.python.org/pep-0257/>`_ (also explained well `here <https://pandas.pydata.org/docs/development/contributing_docstring.html>`_)
* `Shinx Cheat Sheets <https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_

As such, the documentation of a function can look like this:

.. testcode::

    def create_empty_network_data(nodes):
        """
        Function creating connection and distance matrices for defined nodes.

        :param list nodes: list of nodes to create matrices from
        :return: dictionary containing two pandas data frames with a distance and connection matrix respectively
        """
        # initialize return data dict
        data = {}

        # construct connection matrix
        matrix = pd.DataFrame(data=np.full((len(nodes), len(nodes)), 0),
                              index=nodes, columns=nodes)
        data['connection'] = matrix

        # construct distance matrix
        matrix = pd.DataFrame(data=np.full((len(nodes), len(nodes)), 0),
                              index=nodes, columns=nodes)
        data['distance'] = matrix
        return data


Testing new features
----------------------
AdOpT-NET0 comes with a test suite, located in ``.\src\test``. For new features, try to
implement a \
test function in one a respective module (or create a new module). All tests can be executed by \
running ``coverage run -m pytest`` from the terminal.
To check the code coverage of the test, run ``coverage report`` after the test.


Working with GitHub
-----------------------
When you want to develop a new feature of the PyHub, there is a specific procedure you should follow regarding the use \
of GitHub. We follow this procedure to prevent conflicts in the code when multiple people are developing/using the tool.

We use the following guidelines when you are implementing a new feature:

* Open a new branch from the main branch, using an intuitive name. Publish the branch to GitHub.
* Implement the new feature in the branch. Commit the implemented changes to the branch, even if it is an intermediate version, to make sure that local changes (on your personal computer) are also saved online.
* When the new feature is implemented, test it using the testfunctions (described above). If necessary, you can write a new test function to make sure it is tested in future versions.
* Once it is tested, you can make a pull request from your branch. Please make sure that you request a review from at least one other colleague who is working on the tool. You can also add an open issue as label in the development option.
* The colleague evaluates your code and can leave comments or require changes if that is necessary.
* The branch can be merged by the colleague when the code is clear and all required changes are implemented. After merging the branch can be deleted, both locally and on GitHub.
