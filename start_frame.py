# coding:utf-8
import os
import tkinter as tk
from collections import OrderedDict

import yaml


class DataFrame:
    """
        The UI Frame what configure data configuration file.
    """

    def __init__(self, path=None):
        self.top = None

        self.configure_path = path
        if self.configure_path is None:
            self.configure_path = os.path.join(os.path.dirname(__file__), './config/parameters.yaml')

        self.data_parameters = OrderedDict(
            hue=0.1,
            saturation=1.5,
            exposure=1.5,
            jitter_x=0.3,
            jitter_y=0.3,
            flip=1,
            img_size=(1280, 768),
            classes="['person']",
            data_path=''
        )

        self.network_parameters = OrderedDict(
            lr=0.001,
            weight_decay=0.1,
            epochs=100
        )

        if os.path.exists(self.configure_path):
            self.__reset_value()
        self.y = 0

    def run(self):
        """

        :return:
        """
        self.top = tk.Tk()
        self.top.title('Data Frame')
        self.top.geometry('400x600')

        tk.Label(self.top, text="Data:", show=None, font=('Arial', 12)).place(x=0, y=self.y)
        self.y += 24
        self.add_data_configure()

        tk.Label(self.top, text="Net:", show=None, font=('Arial', 12)).place(x=0, y=self.y)
        self.y += 24
        self.add_network_configure()

        b1 = tk.Button(self.top, text='Configure Over', font=('Arial', 12), command=self.get_value)
        b1.place(x=0, y=self.y)
        self.top.mainloop()

    def add_data_configure(self):
        for i, attribute in enumerate(self.data_parameters.keys()):
            y_coordination = self.y + 24 * i + 6
            tk.Label(self.top, text=attribute + ":", show=None, font=('Arial', 12)).place(x=26, y=y_coordination)
            default_value = str(self.data_parameters.get(attribute))
            setattr(self, 'e_{}'.format(attribute),
                    tk.Entry(self.top, textvariable=tk.StringVar(value=default_value), show=None,
                             width=len(default_value) if isinstance(default_value, str) and len(
                                 default_value) is not 0 else 30))
            getattr(self, 'e_{}'.format(attribute)).place(x=108, y=y_coordination)
        self.y += 24 * (i + 1) + 26

    def add_network_configure(self):
        for i, attribute in enumerate(self.network_parameters.keys()):
            y_coordination = self.y + 24 * i + 6
            tk.Label(self.top, text=attribute + ":", show=None, font=('Arial', 12)).place(x=26, y=y_coordination)
            default_value = str(self.network_parameters.get(attribute))
            setattr(self, 'e_{}'.format(attribute),
                    tk.Entry(self.top, textvariable=tk.StringVar(value=default_value), show=None,
                             width=len(default_value) if isinstance(default_value, str) and len(
                                 default_value) is not 0 else 30))
            getattr(self, 'e_{}'.format(attribute)).place(x=26 + 120, y=y_coordination)
        self.y += 24 * (i + 1) + 26

    def get_value(self):
        for attribute in list(self.data_parameters.keys()) + list(self.network_parameters.keys()):
            temp = getattr(self, 'e_{}'.format(attribute)).get()
            if temp is not None:
                setattr(self, '{}'.format(attribute), temp)
        self.top.quit()
        self.do_configure()

    def do_configure(self):
        """

        :return:
        """

        infos = [{}, {}]
        for i, keys in enumerate([list(self.data_parameters.keys()), list(self.network_parameters.keys())]):
            for att in keys:
                try:
                    value = eval(getattr(self, '{}'.format(att)))
                except:
                    value = getattr(self, '{}'.format(att))
                print(att, " :", value)
                infos[i].setdefault(att, value)
        all_attributes = {
            'Data': infos[0],
            'Net': infos[1]
        }
        with open(self.configure_path, 'w', encoding='utf-8') as f:
            yaml.dump(all_attributes, f)

    def __reset_value(self):
        with open(self.configure_path, 'r', encoding='utf-8') as f:
            load_data = yaml.load(f)
        if load_data is not None:
            data_info = load_data.get('Data')
            net_info = load_data.get('Net')
            set_value = [self.data_parameters, self.network_parameters]
            for i, info in enumerate([data_info, net_info]):
                info_keys = info.keys()
                for key in info_keys:
                    # setattr(self.attributes, key, load_data.get(key))
                    set_value[i][key] = str(info.get(key, None))
            # for key in self.attributes.keys():
            #     if key in load_data_keys:
            #         self.attributes[key] = load_data.get(key)


if __name__ == '__main__':
    data_frame = DataFrame()
    data_frame.run()
