from flask_wtf import Form
from wtforms import SubmitField, RadioField, SelectField
from wtforms.fields.html5 import DecimalRangeField

from wtforms import validators, ValidationError

class ContactForm(Form):
    dataset = SelectField('Dataset', choices = [('a', 'Flower'), ('b', 'Noisy Circles'), ('c', 'Noisy Moons'), ('d', 'Blobs'), ('e', 'Gaussian Quantiles')])
    optimizer = SelectField('Optimizer', choices = [('gd', 'Gradient Descent'), ('momentum', 'Gradient Descent with Momentum'), ('adam', 'Adam Optimization')])
    dropout = RadioField('Dropout', choices = [('yr','Yes'),('nr','No')])
    reg_param = DecimalRangeField('Choose lambda ', default=0)
    regularization = RadioField('Regularization', choices = [('yd','Yes'),('nd','No')])
    keep_prob = DecimalRangeField('Choose keep probability ', default=0)
    submit = SubmitField("Submit")
