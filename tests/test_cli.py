from height_mvp.cli import main


def test_cli_loads_default_layout(capsys):
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "Loaded 4 markers" in output
    assert "DICT_4X4_50" in output
