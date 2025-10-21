..   _compression:

Compression and pressure levels
=====================================
Compression is modeled only if **pressure consideration** is enabled in the configuration.
To activate pressure consideration, set the following parameter in ``ConfigModel.json``:

.. code-block:: json

    {
      "performance": {
        "pressure": {
          "pressure_on": { "value": 1 }
        }
      }
    }


Pressure-related calculations are applied **only to components and networks that deal with gas**.
To specify which gas carriers should be modeled with different pressure levels and sequential compression, define them in the configuration file as:

.. code-block:: json

    {
      "performance": {
        "pressure": {
          "pressure_carriers": { "value": ["Gas A", "Gas B"] }
        }
      }
    }

⚠️ **Note:** Currently, compression and multiple pressure levels are supported only for **hydrogen**.

Once the pressure consideration is active, the model will read through all the components their default input and output pressure levels for the chosen gases.

At each node a connection will be created between all pairs of component that share the same gas as output and input.
For each connection a compression will be created as new component (see :ref:`Compressor Class<compressor>`).

Only connections which require an increase of pressure from output component (component that has the gas as output) to input component (component that has the gas as input) will consider the compressor as active.

An active compressor will implicate constrain on energy consumption and investment cost. Constraints for compressors can be found in (see :ref:`Compressor Class<compressor>`).
Constraints for the mass balance can be found in :ref:`Construct Balances<src-code_model-constructing>` under construct_compressor_constrains


