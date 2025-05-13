from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class JobForm(FlaskForm):
    company = StringField("Company", validators=[DataRequired()])
    role = StringField("Role", validators=[DataRequired()])
    location = StringField("Location", validators=[DataRequired()])
    status = StringField("Status", validators=[DataRequired()])
    submit = SubmitField("Add Job")
