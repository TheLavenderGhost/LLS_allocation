import numpy as np
import matplotlib.pyplot as plt
import mplcursors as mc


def incident_matrix(points_nr, edges):
    edges_nr, _ = edges.shape
    matrix = np.zeros((points_nr, edges_nr))
    for e_nr in range(edges_nr):
        matrix[edges[e_nr, 0] - 1, e_nr] = -1
        matrix[edges[e_nr, 1] - 1, e_nr] = 1
    return matrix


def allocation(points_nr, edges, check_points):
    check_points_nr, _ = check_points.shape
    edges_nr, _ = edges.shape
    matrix = incident_matrix(points_nr, edges)
    unknown_points_nr = points_nr - check_points_nr
    matrix_unknown = matrix[:unknown_points_nr]
    matrix_check = matrix[unknown_points_nr:]

    b = np.concatenate((-np.matmul(matrix_check.T, check_points[:, 0]),
                        -np.matmul(matrix_check.T, check_points[:, 1])))
    a = np.zeros((2 * edges_nr, 2 * unknown_points_nr))
    a[:edges_nr, :unknown_points_nr] = matrix_unknown.T
    a[edges_nr:, unknown_points_nr:] = matrix_unknown.T

    lls_answer = np.linalg.lstsq(a, b)
    x_answer = lls_answer[0][:unknown_points_nr]
    y_answer = lls_answer[0][unknown_points_nr:]

    x_check = check_points.T[0]
    y_check = check_points.T[1]

    x_all = np.concatenate((x_answer, x_check))
    y_all = np.concatenate((y_answer, y_check))

    fig, ax = plt.subplots()
    plt.plot(x_check, y_check, "*", color='green', label="punkty kontrolne", zorder=2)
    scatter = ax.scatter(x_answer, y_answer, color='red', label="szukane", zorder=2)
    cursor = mc.cursor(scatter, hover=mc.HoverMode.Transient, highlight=True,
                       highlight_kwargs=dict(fadecolor="yellow"))

    for k in edges:
        x_edge = [x_all[k[0] - 1], x_all[k[1] - 1]]
        y_edge = [y_all[k[0] - 1], y_all[k[1] - 1]]
        plt.plot(x_edge, y_edge, color="black", zorder=1)

    def get_point_nr(target):
        indexes_x = np.where(x_answer == target[0])
        indexes_y = np.where(y_answer == target[1])
        common_indexes = np.intersect1d(indexes_x, indexes_y) + 1
        return ' / '.join(map(str, common_indexes))

    @cursor.connect("add")
    def _(sel):
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
        point_nr = get_point_nr(sel.target)
        sel.annotation.set_text(f"punkt nr: {point_nr}\nx = {sel.target[0]:.4f}\ny = {sel.target[1]:.4f}")

    plt.legend()
    plt.show()
    return


def example_1():
    e = np.array([[6, 3], [7, 3], [7, 5], [8, 5], [9, 1],
                  [9, 2], [9, 4], [10, 1], [10, 4], [2, 4],
                  [2, 5], [3, 4], [3, 5], [4, 5]])
    c = np.array([[1., 5.], [4., 5.], [6., 3.], [3., 0], [0., 3.]])
    allocation(10, e, c)
    return


def example_2():
    e = np.array([[1, 20], [2, 16], [3, 16], [4, 18], [5, 17],
                  [6, 17], [7, 20], [8, 17], [9, 16], [10, 20],
                  [11, 19], [12, 19], [13, 18], [14, 16], [15, 19],
                  [1, 16], [2, 17], [3, 20], [4, 19], [5, 20],
                  [6, 16], [7, 5], [8, 16], [9, 3], [10, 18],
                  [11, 20], [12, 10], [13, 3], [14, 20], [15, 20],
                  [7, 19], [5, 16], [5, 13], [9, 6], [4, 5]])
    c = np.array([[1., 5.], [4., 5.], [6., 3.], [3., 0], [0., 3.]])
    allocation(20, e, c)
    return


if __name__ == "__main__":
    # example_1()
    example_2()
