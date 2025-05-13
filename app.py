from flask import Flask, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from forms import JobForm

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'devkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Job model
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f"<Job {self.company} - {self.role}>"

@app.route("/")
def home():
    return redirect(url_for('list_jobs'))

@app.route("/add", methods=["GET", "POST"])
def add_job():
    form = JobForm()
    if form.validate_on_submit():
        new_job = Job(
            company=form.company.data,
            role=form.role.data,
            location=form.location.data,
            status=form.status.data
        )
        db.session.add(new_job)
        db.session.commit()
        return redirect(url_for('list_jobs'))
    return render_template("add.html", form=form)

@app.route("/jobs")
def list_jobs():
    jobs = Job.query.all()
    return render_template("jobs.html", jobs=jobs)

@app.route("/edit/<int:job_id>", methods=["GET", "POST"])
def edit_job(job_id):
    job = Job.query.get_or_404(job_id)
    form = JobForm(obj=job)
    if form.validate_on_submit():
        form.populate_obj(job)
        db.session.commit()
        return redirect(url_for('list_jobs'))
    return render_template("edit.html", form=form)

@app.route("/delete/<int:job_id>")
def delete_job(job_id):
    job = Job.query.get_or_404(job_id)
    db.session.delete(job)
    db.session.commit()
    return redirect(url_for('list_jobs'))

if __name__ == "__main__":
    app.run(debug=True)
