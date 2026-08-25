# Authentication and database setup

1. Create a Supabase project and run `supabase/migrations/001_initial_schema.sql` in its SQL Editor.
2. In Supabase Auth, enable email confirmation and set the site URL and redirect URL to the Vercel frontend URL.
3. Set frontend environment variables in Vercel: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, and `VITE_API_URL`.
4. Set backend environment variables in Render: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `CORS_ORIGINS` (the exact Vercel URL, no trailing slash).
5. Deploy the backend, then the frontend.

## Clinician access

Patient registration always creates a `patient` profile. Promote verified staff to `doctor` or `clinic_admin` only from the Supabase SQL Editor or a future administrator tool, then assign patients with `clinician_patients`. Do not allow users to choose a clinical role during signup.

```sql
update public.profiles set role = 'doctor' where id = '<clinician-auth-user-id>';
insert into public.clinician_patients (clinician_id, patient_id)
values ('<clinician-auth-user-id>', '<patient-auth-user-id>');
```

This application is screening support software; deployment involving real health data requires a privacy, security, and clinical-governance review.
