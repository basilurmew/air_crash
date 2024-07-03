from flask import Flask, render_template_string, request, render_template
import folium
from table_to_map import map_gen, data_condition
import pandas as pd
import  numpy as np
# create a flask application
app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == "GET":
        """Create a map object"""

        # тут код индуса идет, его не трогать!!!

        df = pd.read_csv('newCrash.csv')

        df1 = df[np.isnan(df["latitude"]) != True]

        mapObj = map_gen(df1)

        # render the map object
        mapObj.get_root().render()

        # derive the script and style tags to be rendered in HTML head
        header = mapObj.get_root().header.render()

        # derive the div container to be rendered in the HTML body
        body_html = mapObj.get_root().html.render()

        # derive the JavaScript to be rendered in the HTML body
        script = mapObj.get_root().script.render()

        # return a web page with folium map components embeded in it. You can also use render_template.
        return render_template("map.html",
            header=header,
            body_html=body_html,
            script=script,
        )
    else:
        min_year = int(request.form.get('min_year'))
        max_year = int(request.form.get('max_year'))
        df = pd.read_csv('newCrash.csv')
        df1 = df[np.isnan(df["latitude"]) != True]
        date_df = data_condition(df1, min_year,max_year)
        mapObj = map_gen(date_df)
        mapObj.get_root().render()
        header = mapObj.get_root().header.render()
        body_html = mapObj.get_root().html.render()
        script = mapObj.get_root().script.render()
        return render_template("map.html",
                               header=header,
                               body_html=body_html,
                               script=script,
                               )


if __name__ == "__main__":
    app.run()