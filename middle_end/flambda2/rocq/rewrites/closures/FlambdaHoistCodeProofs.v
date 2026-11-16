(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaHoistCodeSpec.v. Nothing here
   needs review.

   Two things are kept along the run of a unit that passes the checks.
   The closures that its terms make are of code that is bound, as more
   and more code is bound ([terms_in_scope]), so the closures that its
   values hold are (FlambdaSemanticsProofs.step_values_all). And no
   code id is bound twice (the invariant of
   FlambdaWellFormedProofs.v). The unit and the unit with its code
   first are then run side by side, in step, with the same values: the
   second state is the first with the bindings of code taken out of its
   terms and all the code bound ([hoist_state]). A call finds its code
   in both, because the closure called is of code that the first run
   has bound; a group of static bindings binds its symbols in both,
   because the first run does not stop on the code of the group.

   The two runs may start from a continuation environment of what may
   follow a unit (FlambdaObserversSpec.observers), the empty one for
   the statements of a unit run on its own. What may follow a unit
   binds no code, so the second state has it as it is
   ([observers_hoist]); it names no code id, so the closures that it
   makes are of code that is bound, whatever is bound
   ([observers_in_scope]); and it keeps no value. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNormaliseSpec FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaHeapProofs FlambdaUnusedSlotsProofs
  FlambdaImportsSpec FlambdaStaticsInScopeSpec FlambdaObserversSpec
  FlambdaUnitFromProofs FlambdaHoistCodeSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* More code bound                                                          *)
(* ======================================================================== *)

Definition bound_le (bound bound' : code_id -> bool) : Prop :=
  ∀ id : code_id, bound id = true -> bound' id = true.

Lemma bound_le_with (ids : list code_id) (bound bound' : code_id -> bool) :
  bound_le bound bound' ->
  bound_le (bound_with ids bound) (bound_with ids bound').
Proof.
  intros Hle id. unfold bound_with. rewrite !orb_true_iff.
  intros [H | H]; [ left; exact H | right; exact (Hle id H) ].
Qed.

Lemma decls_bound_le (bound bound' : code_id -> bool)
    (decls : list (function_slot * code_id)) :
  bound_le bound bound' ->
  decls_bound bound decls = true -> decls_bound bound' decls = true.
Proof.
  intros Hle. unfold decls_bound. rewrite !forallb_forall.
  intros H decl Hin. exact (Hle _ (H decl Hin)).
Qed.

Lemma named_closures_bound_le (bound bound' : code_id -> bool) (d : named) :
  bound_le bound bound' ->
  named_closures_bound bound d = true -> named_closures_bound bound' d = true.
Proof.
  intros Hle. destruct d; simpl; try (intros _; reflexivity).
  - apply Hle.
  - apply decls_bound_le, Hle.
Qed.

Lemma const_closures_bound_le (bound bound' : code_id -> bool)
    (const : static_const) :
  bound_le bound bound' ->
  const_closures_bound bound const = true ->
  const_closures_bound bound' const = true.
Proof.
  intros Hle. destruct const; simpl; try (intros _; reflexivity).
  - apply Hle.
  - apply decls_bound_le, Hle.
Qed.

Fixpoint closures_in_scope_le (bound bound' : code_id -> bool) (e : expr)
    {struct e} :
  bound_le bound bound' ->
  closures_in_scope bound e = true -> closures_in_scope bound' e = true
with handler_closures_in_scope_le (bound bound' : code_id -> bool)
    (h : continuation_handler) {struct h} :
  bound_le bound bound' ->
  handler_closures_in_scope bound h = true ->
  handler_closures_in_scope bound' h = true
with code_closures_in_scope_le (bound bound' : code_id -> bool) (c : code)
    {struct c} :
  bound_le bound bound' ->
  code_closures_in_scope bound c = true ->
  code_closures_in_scope bound' c = true.
Proof.
  - intros Hle.
    destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [closures_in_scope]; try (intros _; reflexivity).
    + intros H. apply andb_true_iff in H as [Hd Hbody].
      rewrite (named_closures_bound_le bound bound' d Hle Hd),
        (closures_in_scope_le bound bound' body Hle Hbody).
      reflexivity.
    + intros H. apply andb_true_iff in H as [Hbindings Hbody].
      pose proof
        (bound_le_with (flat_map code_ids_of_binding bindings) bound bound'
           Hle)
        as Hle'.
      rewrite (closures_in_scope_le _ _ body Hle' Hbody), andb_true_r.
      clear Hbody.
      set (b := bound_with (flat_map code_ids_of_binding bindings) bound)
        in *.
      set (b' := bound_with (flat_map code_ids_of_binding bindings) bound')
        in *.
      clearbody b b'.
      induction bindings as [ | binding bindings IH]; [ reflexivity | ].
      cbn [forallb] in Hbindings |- *.
      apply andb_true_iff in Hbindings as [Hbinding Hrest].
      rewrite (IH Hrest), andb_true_r.
      destruct binding as [id c | id | sym const];
        cbn [binding_closures_in_scope] in Hbinding |- *.
      * exact (code_closures_in_scope_le b b' c Hle' Hbinding).
      * reflexivity.
      * exact (const_closures_bound_le b b' const Hle' Hbinding).
    + intros H. apply andb_true_iff in H as [Hh Hbody].
      rewrite (handler_closures_in_scope_le bound bound' h Hle Hh),
        (closures_in_scope_le bound bound' body Hle Hbody).
      reflexivity.
    + intros H. apply andb_true_iff in H as [Hhandlers Hbody].
      rewrite (closures_in_scope_le bound bound' body Hle Hbody),
        andb_true_r.
      clear Hbody.
      induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
      cbn [forallb] in Hhandlers |- *.
      apply andb_true_iff in Hhandlers as [Hh Hrest].
      rewrite (IH Hrest), andb_true_r.
      exact (handler_closures_in_scope_le bound bound' h Hle Hh).
  - intros Hle. destruct h as [params metadata handler].
    cbn [handler_closures_in_scope].
    apply closures_in_scope_le, Hle.
  - intros Hle. destruct c as [params my_closure ret exn metadata body].
    cbn [code_closures_in_scope].
    apply closures_in_scope_le, Hle.
Qed.

Lemma binding_closures_in_scope_le (bound bound' : code_id -> bool)
    (binding : static_binding) :
  bound_le bound bound' ->
  binding_closures_in_scope bound binding = true ->
  binding_closures_in_scope bound' binding = true.
Proof.
  intros Hle. destruct binding as [id c | id | sym const]; simpl.
  - apply code_closures_in_scope_le, Hle.
  - intros _. reflexivity.
  - apply const_closures_bound_le, Hle.
Qed.

Lemma handlers_closures_in_scope_le (bound bound' : code_id -> bool)
    (handlers : list (continuation * continuation_handler)) :
  bound_le bound bound' ->
  forallb (fun '(_, h) => handler_closures_in_scope bound h) handlers
  = true ->
  forallb (fun '(_, h) => handler_closures_in_scope bound' h) handlers
  = true.
Proof.
  intros Hle. rewrite !forallb_forall. intros H [k h] Hin.
  exact (handler_closures_in_scope_le bound bound' h Hle (H (k, h) Hin)).
Qed.

Lemma value_closures_bound_le (bound bound' : code_id -> bool) (v : value) :
  bound_le bound bound' ->
  value_closures_bound bound v = true -> value_closures_bound bound' v = true.
Proof.
  intros Hle.
  induction v
    as [ c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs ]
    using value_induction;
    simpl; try (intros _; reflexivity).
  - intros H. apply andb_true_iff in H as [Hcode Hslots].
    rewrite (Hle code Hcode). simpl.
    rewrite forallb_forall in Hslots |- *. rewrite Forall_forall in IH.
    intros slot Hin. exact (IH slot Hin (Hslots slot Hin)).
  - intros H. apply andb_true_iff in H as [Hdecls Hslots].
    rewrite (decls_bound_le bound bound' decls Hle Hdecls). simpl.
    rewrite forallb_forall in Hslots |- *. rewrite Forall_forall in IH.
    intros vslot Hin. exact (IH vslot Hin (Hslots vslot Hin)).
  - intros H. rewrite forallb_forall in H |- *. rewrite Forall_forall in IH.
    intros field Hin. exact (IH field Hin (H field Hin)).
  - intros H. rewrite forallb_forall in H |- *. rewrite Forall_forall in IH.
    intros element Hin. exact (IH element Hin (H element Hin)).
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    rewrite (IHcallee Hcallee). simpl.
    rewrite forallb_forall in Hargs |- *. rewrite Forall_forall in IHargs.
    intros arg Hin. exact (IHargs arg Hin (Hargs arg Hin)).
Qed.

Lemma slots_closures_bound_le (bound bound' : code_id -> bool)
    (slots : list (value_slot * value)) :
  bound_le bound bound' ->
  forallb (fun slot => value_closures_bound bound (snd slot)) slots = true ->
  forallb (fun slot => value_closures_bound bound' (snd slot)) slots = true.
Proof.
  intros Hle. rewrite !forallb_forall. intros H slot Hin.
  exact (value_closures_bound_le bound bound' (snd slot) Hle (H slot Hin)).
Qed.

Lemma allocated_closures_bound_le (bound bound' : code_id -> bool)
    (allocated : statically_allocated_value) :
  bound_le bound bound' ->
  allocated_closures_bound bound allocated = true ->
  allocated_closures_bound bound' allocated = true.
Proof.
  intros Hle.
  destruct allocated as [code slots | slot syms decls slots | tag fields
                        | contents | bytes | elements | c];
    simpl.
  - intros H. apply andb_true_iff in H as [Hcode Hslots].
    rewrite (Hle code Hcode), (slots_closures_bound_le _ _ slots Hle Hslots).
    reflexivity.
  - intros H. apply andb_true_iff in H as [Hdecls Hslots].
    rewrite (decls_bound_le _ _ decls Hle Hdecls),
      (slots_closures_bound_le _ _ slots Hle Hslots).
    reflexivity.
  - rewrite !forallb_forall. intros H field Hin.
    exact (value_closures_bound_le bound bound' field Hle (H field Hin)).
  - apply value_closures_bound_le, Hle.
  - intros _. reflexivity.
  - rewrite !forallb_forall. intros H element Hin.
    exact (value_closures_bound_le bound bound' element Hle (H element Hin)).
  - intros _. reflexivity.
Qed.

(* What a group of static bindings binds of code is bound after it. *)
Lemma eval_static_bindings_code_bound (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    eval_static_bindings vars bindings statics = Some statics' ->
    bound_le
      (bound_with (flat_map code_ids_of_binding bindings)
         (code_id_bound statics))
      (code_id_bound statics').
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros [= <-] n Hn. exact Hn.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval n Hn. apply (IH _ _ Heval n).
    unfold bound_with in Hn |- *. rewrite among_cons in Hn.
    unfold code_id_bound in Hn |- *. simpl.
    destruct (Pos.eqb n id); [ apply orb_true_r | exact Hn ].
  - apply IH.
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    intros Heval n Hn. exact (IH _ _ Heval n Hn).
Qed.

(* Code that is bound after a group was bound before it, or is bound by
   the group. *)
Lemma eval_static_bindings_code_in (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (id : code_id) (c : code),
    eval_static_bindings vars bindings statics = Some statics' ->
    lookup id (static_code statics') = Some c ->
    lookup id (static_code statics) = Some c ∨ Bind_code id c ∈ bindings.
Proof.
  induction bindings as [ | [id' c' | id' | sym const] bindings IH];
    intros statics statics' id c; simpl.
  - intros [= <-] Hlookup. left. exact Hlookup.
  - destruct (lookup id' (static_code statics)); [ discriminate | ].
    intros Heval Hlookup.
    destruct (IH _ _ id c Heval Hlookup) as [Hold | Hin];
      [ | right; right; exact Hin ].
    simpl in Hold. destruct (Pos.eqb_spec id id') as [-> | _];
      [ injection Hold as <-; right; left; reflexivity | left; exact Hold ].
  - intros Heval Hlookup.
    destruct (IH _ _ id c Heval Hlookup) as [Hold | Hin];
      [ left; exact Hold | right; right; exact Hin ].
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    intros Heval Hlookup.
    destruct (IH _ _ id c Heval Hlookup) as [Hold | Hin];
      [ left; exact Hold | right; right; exact Hin ].
Qed.

(* ======================================================================== *)
(* The closures in values                                                   *)
(* ======================================================================== *)

Section Values.

Variable bound : code_id -> bool.

Definition value_ok (v : value) : Prop := value_closures_bound bound v = true.

Lemma value_ok_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_ok (Value_partial_application callee args args_arity)
  <-> value_ok callee ∧ Forall value_ok args.
Proof.
  unfold value_ok. simpl. rewrite andb_true_iff, forallb_Forall.
  reflexivity.
Qed.

Lemma not_a_variable_ok (s : simple) (v : value) :
  eval_simple [] s = Some v -> value_ok v.
Proof.
  destruct s as [x | sym | c]; simpl;
    [ discriminate | intros [= <-]; reflexivity
    | intros [= <-]; reflexivity ].
Qed.

Lemma eval_simple_ok (vars : env) (s : simple) (v : value) :
  env_values_all value_ok vars -> eval_simple vars s = Some v -> value_ok v.
Proof. exact (eval_simple_values_all value_ok vars s v not_a_variable_ok). Qed.

Lemma eval_simples_ok (vars : env) (args : list simple) (vs : list value) :
  env_values_all value_ok vars ->
  eval_simples vars args = Some vs -> Forall value_ok vs.
Proof.
  intros Hvars.
  exact (eval_simples_values_all value_ok vars args not_a_variable_ok Hvars
           vs).
Qed.

Lemma eval_value_slots_ok (vars : env) :
  env_values_all value_ok vars ->
  ∀ (slots : list (value_slot * simple)) (vs : list (value_slot * value)),
    eval_value_slots vars slots = Some vs ->
    forallb (fun slot => value_closures_bound bound (snd slot)) vs = true.
Proof.
  intros Hvars. induction slots as [ | [slot s] slots IH]; simpl; intros vs.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs' | ]; simpl;
      [ | discriminate ].
    intros [= <-]. simpl. rewrite (IH vs' eq_refl), andb_true_r.
    exact (eval_simple_ok vars s v Hvars Hs).
Qed.

Lemma eval_static_const_ok (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  const_closures_bound bound const = true ->
  env_values_all value_ok vars ->
  eval_static_const vars const = Some allocated ->
  allocated_closures_bound bound allocated = true.
Proof.
  intros Hconst Hvars.
  destruct const as [code slots | slot syms decls slots | tag fields
                    | contents | bytes | elements];
    simpl in Hconst |- *.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. simpl.
    rewrite Hconst, (eval_value_slots_ok vars Hvars slots vs Hslots).
    reflexivity.
  - destruct (lookup slot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. simpl.
    rewrite Hconst, (eval_value_slots_ok vars Hvars slots vs Hslots).
    reflexivity.
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields; simpl;
      [ | discriminate ].
    intros [= <-]. simpl. apply forallb_Forall.
    exact (eval_simples_ok vars fields vs Hvars Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents; simpl;
      [ | discriminate ].
    intros [= <-]. exact (eval_simple_ok vars contents v Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements; simpl;
      [ | discriminate ].
    intros [= <-]. simpl. apply forallb_Forall.
    exact (eval_simples_ok vars elements vs Hvars Helements).
Qed.

(* A group whose constants are of code that is bound keeps what the
   symbols are bound to so. *)
Lemma eval_static_bindings_ok (vars : env) :
  env_values_all value_ok vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb (binding_closures_in_scope bound) bindings = true ->
    Forall
      (fun binding => allocated_closures_bound bound (snd binding) = true)
      (static_values statics) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    Forall
      (fun binding => allocated_closures_bound bound (snd binding) = true)
      (static_values statics').
Proof.
  intros Hvars.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; cbn [forallb eval_static_bindings].
  - intros _ H [= <-]. exact H.
  - intros Hb H. apply andb_true_iff in Hb as [_ Hrest].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). exact H.
  - intros Hb H. apply andb_true_iff in Hb as [_ Hrest].
    exact (IH _ _ Hrest H).
  - intros Hb H. apply andb_true_iff in Hb as [Hconst Hrest].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Heval;
      simpl; [ | discriminate ].
    apply (IH _ _ Hrest). simpl. apply Forall_cons; [ | exact H ].
    exact (eval_static_const_ok vars const allocated Hconst Hvars Heval).
Qed.

(* What a defining expression reads of the static environment. *)
Variable statics : static_env.

Hypothesis statics_ok :
  Forall (fun binding => allocated_closures_bound bound (snd binding) = true)
    (static_values statics).

Lemma allocated_ok (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values statics) = Some allocated ->
  allocated_closures_bound bound allocated = true.
Proof.
  intros Hlookup. apply lookup_in in Hlookup.
  rewrite Forall_forall in statics_ok.
  exact (statics_ok (s, allocated) Hlookup).
Qed.

Lemma value_ok_of_numbers : holds_of_numbers value_ok.
Proof. intros c. split; reflexivity. Qed.

Lemma value_ok_of_blocks : holds_of_blocks value_ok.
Proof.
  intros tag fields. unfold value_ok. simpl. rewrite forallb_forall.
  reflexivity.
Qed.

Lemma value_ok_of_arrays : holds_of_arrays value_ok.
Proof.
  intros elements. unfold value_ok. simpl. rewrite forallb_forall.
  reflexivity.
Qed.

Lemma value_ok_in_static_blocks : holds_in_static_blocks statics value_ok.
Proof.
  intros s tag fields _ Hlookup v Hin.
  pose proof (allocated_ok s _ Hlookup) as Hfields. simpl in Hfields.
  rewrite forallb_forall in Hfields. exact (Hfields v Hin).
Qed.

Lemma value_ok_in_static_arrays : holds_in_static_arrays statics value_ok.
Proof.
  intros s elements _ Hlookup v Hin.
  pose proof (allocated_ok s _ Hlookup) as Helements. simpl in Helements.
  rewrite forallb_forall in Helements. exact (Helements v Hin).
Qed.

(* The fields of a tuple whose closures are of bound code are so too:
   they are read by a primitive. *)
Lemma tuple_fields_ok (size : nat) (tuple : value) (fields : list value) :
  value_ok tuple ->
  tuple_fields statics size tuple = Some fields ->
  Forall value_ok fields.
Proof.
  intros Htuple Hfields.
  apply (tuple_fields_forall _ statics size tuple fields);
    [ | exact Hfields ].
  intros field v Hv.
  refine
    (eval_prim_preserves value_ok statics _ [tuple] v value_ok_of_numbers
       value_ok_of_blocks value_ok_of_arrays value_ok_in_static_blocks
       value_ok_in_static_arrays _ Hv).
  intros w [<- | [ ]]. exact Htuple.
Qed.

Lemma closure_in_set_ok (v : value) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  value_ok v ->
  closure_in_set statics v = Some (slot, decls, slots) ->
  decls_bound bound decls = true
  ∧ forallb (fun vslot => value_closures_bound bound (snd vslot)) slots
    = true.
Proof.
  unfold value_ok. intros Hv.
  destruct v
    as [ c | code vslots | vslot vdecls vslots | s | tag fields
       | elements | c | l | callee args args_arity ];
    simpl in Hv |- *; try discriminate.
  - intros [= <- <- <-]. apply andb_true_iff in Hv as [Hcode Hslots].
    split; [ | exact Hslots ]. unfold decls_bound. simpl. rewrite Hcode.
    reflexivity.
  - destruct (lookup vslot vdecls); simpl; [ | discriminate ].
    intros [= <- <- <-]. apply andb_true_iff in Hv. exact Hv.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    pose proof (allocated_ok s allocated Hlookup) as Hallocated.
    destruct allocated
      as [code aslots | aslot syms adecls aslots | tag fields | contents
         | bytes | elements | ];
      simpl in Hallocated |- *; try discriminate.
    + intros [= <- <- <-].
      apply andb_true_iff in Hallocated as [Hcode Hslots].
      split; [ | exact Hslots ]. unfold decls_bound. simpl. rewrite Hcode.
      reflexivity.
    + destruct (lookup aslot adecls); simpl; [ | discriminate ].
      intros [= <- <- <-]. apply andb_true_iff in Hallocated.
      exact Hallocated.
Qed.

(* A closure that is called is of code that is bound. *)
Lemma closure_of_ok (v : value) (called : code_id)
    (slots : list (value_slot * value)) :
  value_ok v ->
  closure_of statics v = Some (called, slots) ->
  bound called = true
  ∧ forallb (fun vslot => value_closures_bound bound (snd vslot)) slots
    = true.
Proof.
  intros Hv Hclosure.
  destruct (proj1 (closure_in_set_of statics v called slots) Hclosure)
    as (slot & decls & Hin_set & Hlookup).
  destruct (closure_in_set_ok v slot decls slots Hv Hin_set)
    as [Hdecls Hslots].
  split; [ | exact Hslots ].
  unfold decls_bound in Hdecls. rewrite forallb_forall in Hdecls.
  exact (Hdecls (slot, called) (lookup_in _ _ _ Hlookup)).
Qed.

Lemma project_function_slot_ok (move_from move_to : function_slot)
    (c v : value) :
  value_ok c ->
  project_function_slot statics move_from move_to c = Some v ->
  value_ok v.
Proof.
  unfold value_ok, project_function_slot. intros Hc.
  destruct c
    as [ k | code vslots | vslot vdecls vslots | s | tag fields
       | elements | k | l | callee args args_arity ];
    try discriminate.
  - destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hc.
  - destruct (Pos.eqb vslot move_from); [ | discriminate ].
    destruct (lookup vslot vdecls); simpl; [ | discriminate ].
    destruct (lookup move_to vdecls); simpl; [ | discriminate ].
    intros [= <-]. exact Hc.
  - destruct (lookup s (static_values statics))
      as [[code aslots | aslot syms adecls aslots | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. reflexivity.
    + destruct (Pos.eqb aslot move_from); [ | discriminate ].
      destruct (lookup aslot adecls); simpl; [ | discriminate ].
      destruct (lookup move_to adecls); simpl; [ | discriminate ].
      destruct (lookup move_to syms); simpl; [ | discriminate ].
      intros [= <-]. reflexivity.
Qed.

Lemma eval_named_ok (vars : env) (d : named) (v : value) :
  named_closures_bound bound d = true ->
  env_values_all value_ok vars ->
  eval_named statics vars d = Some v ->
  value_ok v.
Proof.
  intros Hd Hvars.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | project_from slot closure | move_from move_to closure ];
    simpl in Hd |- *.
  - exact (eval_simple_ok vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs; simpl;
      [ | discriminate ].
    intros Hprim.
    apply (eval_prim_preserves value_ok statics p vs v value_ok_of_numbers
             value_ok_of_blocks value_ok_of_arrays value_ok_in_static_blocks
             value_ok_in_static_arrays);
      [ | exact Hprim ].
    pose proof (eval_simples_ok vars args vs Hvars Hargs) as Hvs.
    rewrite Forall_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. unfold value_ok. simpl.
    rewrite Hd, (eval_value_slots_ok vars Hvars slots vs Hslots).
    reflexivity.
  - destruct (lookup slot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. unfold value_ok. simpl.
    rewrite Hd, (eval_value_slots_ok vars Hvars slots vs Hslots).
    reflexivity.
  - destruct (eval_simple vars closure) as [c | ] eqn:Hc; simpl;
      [ | discriminate ].
    destruct (closure_of statics c) as [[code slot_values] | ] eqn:Hclosure;
      simpl; [ | discriminate ].
    intros Hlookup.
    destruct (closure_of_ok c code slot_values
                (eval_simple_ok vars closure c Hvars Hc) Hclosure)
      as [_ Hslots].
    rewrite forallb_forall in Hslots.
    exact (Hslots (slot, v) (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [c | ] eqn:Hc; simpl;
      [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set statics c) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      simpl; [ | discriminate ].
    destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_ok c at_slot decls slot_values
                (eval_simple_ok vars closure c Hvars Hc) Hin_set)
      as [_ Hslots].
    rewrite forallb_forall in Hslots.
    exact (Hslots (slot, v) (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [c | ] eqn:Hc; simpl;
      [ | discriminate ].
    exact (project_function_slot_ok move_from move_to c v
             (eval_simple_ok vars closure c Hvars Hc)).
Qed.

(* The same of [exec_named], with the values that the heap holds. *)
Lemma exec_named_ok (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (d : named) (v : value) :
  named_closures_bound bound d = true ->
  env_values_all value_ok vars ->
  heap_values_all value_ok hp ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  value_ok v ∧ heap_values_all value_ok hp'.
Proof.
  intros Hd Hvars Hheap.
  apply (exec_named_values_all value_ok statics hp hp' rd rd' vars d v
           value_ok_of_numbers value_ok_of_blocks value_ok_of_arrays
           value_ok_in_static_blocks value_ok_in_static_arrays
           (fun _ => eq_refl) Hheap).
  - intros p args vs _ Hvs.
    pose proof (eval_simples_ok vars args vs Hvars Hvs) as Hall.
    rewrite Forall_forall in Hall. exact Hall.
  - intros _ w. exact (eval_named_ok vars d w Hd Hvars).
Qed.

Lemma called_closure_ok (callee : value) :
  value_ok callee -> value_ok (called_closure callee).
Proof.
  unfold value_ok. destruct callee; simpl; try (intros H; exact H).
  intros H. apply andb_true_iff in H. exact (proj1 H).
Qed.

End Values.

(* ======================================================================== *)
(* The terms of a state make closures of code that is bound                 *)
(* ======================================================================== *)

(* The handlers that a continuation keeps. *)
Fixpoint cont_in_scope (bound : code_id -> bool) (c : cont_value) : Prop :=
  match c with
  | Outer _ => True
  | Closure h _ conts recursive =>
    handler_closures_in_scope bound h = true
    ∧ forallb (fun '(_, h) => handler_closures_in_scope bound h) recursive
      = true
    ∧ fold_right (fun kc acc => cont_in_scope bound (snd kc) ∧ acc) True conts
  | Extra_args c _ => cont_in_scope bound c
  | Over_application _ _ _ return_to exn_to =>
    cont_in_scope bound return_to ∧ cont_in_scope bound exn_to
  end.

Definition conts_in_scope (bound : code_id -> bool) (conts : cont_env)
    : Prop :=
  fold_right (fun kc acc => cont_in_scope bound (snd kc) ∧ acc) True conts.

Fixpoint cont_in_scope_le (bound bound' : code_id -> bool)
    (Hle : bound_le bound bound') (c : cont_value) {struct c} :
  cont_in_scope bound c -> cont_in_scope bound' c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl; [ | auto | apply cont_in_scope_le, Hle | ].
  - intros (Hh & Hrec & Hconts).
    refine
      (conj (handler_closures_in_scope_le bound bound' h Hle Hh)
         (conj (handlers_closures_in_scope_le bound bound' recursive Hle Hrec)
            _)).
    induction conts as [ | [k c] conts IH]; simpl; [ exact I | ].
    destruct Hconts as [Hc Hrest].
    split;
      [ exact (cont_in_scope_le bound bound' Hle c Hc) | exact (IH Hrest) ].
  - intros [Hreturn Hexn].
    exact (conj (cont_in_scope_le bound bound' Hle return_to Hreturn)
             (cont_in_scope_le bound bound' Hle exn_to Hexn)).
Qed.

Lemma conts_in_scope_le (bound bound' : code_id -> bool) (conts : cont_env) :
  bound_le bound bound' ->
  conts_in_scope bound conts -> conts_in_scope bound' conts.
Proof.
  intros Hle. unfold conts_in_scope.
  induction conts as [ | [k c] conts IH]; simpl; [ auto | ].
  intros [Hc Hrest].
  split; [ exact (cont_in_scope_le bound bound' Hle c Hc) | exact (IH Hrest) ].
Qed.

Lemma conts_in_scope_app (bound : code_id -> bool) (conts conts' : cont_env) :
  conts_in_scope bound conts -> conts_in_scope bound conts' ->
  conts_in_scope bound (conts ++ conts').
Proof.
  unfold conts_in_scope.
  induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_in_scope (bound : code_id -> bool) (k : continuation)
    (conts : cont_env) :
  conts_in_scope bound conts -> cont_in_scope bound (resolve k conts).
Proof.
  unfold resolve, conts_in_scope.
  induction conts as [ | [k' c] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma rec_closures_in_scope (bound : code_id -> bool) (vars : env)
    (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_closures_in_scope bound h) all = true ->
  conts_in_scope bound conts ->
  forallb (fun '(_, h) => handler_closures_in_scope bound h) handlers
  = true ->
  conts_in_scope bound
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hall Hconts. unfold conts_in_scope.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros H. apply andb_true_iff in H as [Hh Hrest].
  split; [ | exact (IH Hrest) ].
  split; [ exact Hh | ]. split; [ exact Hall | exact Hconts ].
Qed.

Lemma with_extra_args_in_scope (bound : code_id -> bool) (c : cont_value)
    (extra : list value) :
  cont_in_scope bound c -> cont_in_scope bound (with_extra_args c extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

Definition code_in_scope (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    code_closures_in_scope (code_id_bound statics) c = true.

(* What may follow a unit names no code id, so the closures that it
   makes are of code that is bound, whatever is bound: it makes none
   of the code of a function. *)
Lemma named_no_code_closures_bound (bound : code_id -> bool)
    (syms : list symbol) (d : named) :
  named_statics_in_scope syms [] d = true ->
  named_closures_bound bound d = true.
Proof.
  destruct d as [ | | code slots | slot decls slots | | | ]; simpl;
    try reflexivity.
  - discriminate.
  - intros H. apply andb_true_iff in H as [Hdecls _].
    destruct decls as [ | decl decls]; [ reflexivity | ].
    simpl in Hdecls. discriminate Hdecls.
Qed.

Fixpoint observer_closures_in_scope (bound : code_id -> bool) (e : expr)
    {struct e} :
  binds_no_static e = true ->
  expr_statics_in_scope [] [] e = true ->
  closures_in_scope bound e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _ _; reflexivity).
  - intros Hno H. apply andb_true_iff in H as [Hd Hbody].
    rewrite (named_no_code_closures_bound bound [] d Hd).
    exact (observer_closures_in_scope bound body Hno Hbody).
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros Hno H.
    apply andb_true_iff in Hno as [Hno_h Hno_b].
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hh.
    simpl in Hh.
    rewrite (observer_closures_in_scope bound handler Hno_h Hh),
      (observer_closures_in_scope bound body Hno_b Hb).
    reflexivity.
  - intros Hno H. apply andb_true_iff in Hno as [Hno_hs Hno_b].
    apply andb_true_iff in H as [Hhs Hb].
    rewrite (observer_closures_in_scope bound body Hno_b Hb), andb_true_r.
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hhs.
    simpl in Hhs.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hno_hs, Hhs.
    apply andb_true_iff in Hno_hs as [Hno_h Hno_rest].
    apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (observer_closures_in_scope bound handler Hno_h Hh).
    exact (IH Hno_rest Hrest).
Qed.

Lemma observers_in_scope (bound : code_id -> bool) (slots : list value_slot)
    (conts : cont_env) :
  observers slots conts -> conts_in_scope bound conts.
Proof.
  unfold observers, conts_in_scope.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct c
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & Hscope & _).
  split; [ exact (observer_closures_in_scope bound body Hno Hscope) | ].
  split; [ reflexivity | exact I ].
Qed.

Definition terms_in_scope (s : state) : Prop :=
  closures_in_scope (code_id_bound (state_statics s)) (state_expr s) = true
  ∧ conts_in_scope (code_id_bound (state_statics s)) (state_conts s)
  ∧ code_in_scope (state_statics s).

Lemma terms_in_scope_over (bound : code_id -> bool) (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_in_scope bound
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_in_scope bound return_to ∧ cont_in_scope bound exn_to.
Proof. reflexivity. Qed.

Lemma terms_in_scope_code (statics : static_env) (hp : heap)
    (rd : read_record) :
  code_in_scope statics ->
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    cont_in_scope (code_id_bound statics) return_to ->
    cont_in_scope (code_id_bound statics) exn_to ->
    terms_in_scope
      (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).
Proof.
  intros S called c callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold terms_in_scope. simpl_state.
  pose proof (S called c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body].
  exact (conj Hc (conj (conj Hreturn (conj Hexn I)) S)).
Qed.

Lemma terms_in_scope_handler (statics : static_env) (hp : heap)
    (rd : read_record) :
  code_in_scope statics ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env),
    cont_in_scope (code_id_bound statics)
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    terms_in_scope
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S params metadata handler def_vars def_conts recursive handler_vars
    Cr.
  unfold terms_in_scope. simpl_state. simpl in Cr.
  destruct Cr as (Hh & Hrec & Cdef).
  refine (conj Hh (conj (conts_in_scope_app _ _ _ _ Cdef) S)).
  exact (rec_closures_in_scope _ def_vars def_conts recursive recursive Hrec
           Cdef Hrec).
Qed.

Lemma terms_in_scope_step (s s' : state) :
  terms_in_scope s -> step s = Step s' -> terms_in_scope s'.
Proof.
  intros (P & C & S) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep; cbn [closures_in_scope] in P.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold terms_in_scope. simpl_state.
    apply andb_true_iff in P as [_ Pbody].
    exact (conj Pbody (conj C S)).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_in_scope. simpl_state.
    apply andb_true_iff in P as [Pbindings Pbody].
    pose proof (eval_static_bindings_code_bound vars bindings statics statics'
      Heval) as Hle.
    assert (Hle0 : bound_le (code_id_bound statics) (code_id_bound statics')).
    { intros id Hid. apply Hle. unfold bound_with. rewrite Hid.
      apply orb_true_r. }
    refine
      (conj (closures_in_scope_le _ _ body Hle Pbody)
         (conj (conts_in_scope_le _ _ conts Hle0 C) _)).
    intros id c Hlookup.
    destruct (eval_static_bindings_code_in vars bindings statics statics' id c
                Heval Hlookup)
      as [Hold | Hin].
    + exact (code_closures_in_scope_le _ _ c Hle0 (S id c Hold)).
    + rewrite forallb_forall in Pbindings.
      exact (code_closures_in_scope_le _ _ c Hle
               (Pbindings (Bind_code id c) Hin)).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold terms_in_scope. simpl_state.
    apply andb_true_iff in P as [Hh Pbody].
    refine (conj Pbody (conj (conj _ C) S)).
    simpl. exact (conj Hh (conj eq_refl C)).
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_in_scope. simpl_state.
    apply andb_true_iff in P as [Phandlers Pbody].
    refine (conj Pbody (conj (conts_in_scope_app _ _ _ _ C) S)).
    exact (rec_closures_in_scope _ vars conts handlers handlers Phandlers C
             Phandlers).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands statics _ rd_ (cont_in_scope (code_id_bound statics))
         terms_in_scope (fun c extra H => H)
         (terms_in_scope_over (code_id_bound statics))
         (terms_in_scope_code statics _ rd_ S)
           (terms_in_scope_handler statics _ rd_ S)
         _ _ _ _ _ _ _ s' (resolve_in_scope _ _ conts C) _ Hstep).
    apply with_extra_args_in_scope, resolve_in_scope, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands statics _ rd_
         (cont_in_scope (code_id_bound statics))
         terms_in_scope (fun c extra H => H)
         (terms_in_scope_over (code_id_bound statics))
         (terms_in_scope_code statics _ rd_ S)
           (terms_in_scope_handler statics _ rd_ S)
         _ vs s' (resolve_in_scope _ _ conts C) Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    injection Hstep as <-. unfold terms_in_scope. simpl_state.
    exact (conj eq_refl (conj C S)).
  - discriminate.
Qed.

(* The closures in the values of a state are of code that is bound, as
   long as its terms make closures of such code only: what
   FlambdaSemanticsSpec.step_values_all_statement asks of the state. *)
Lemma closures_bound_partial (statics : static_env) (callee : value)
    (args : list value) (args_arity : arity) :
  value_ok (code_id_bound statics)
    (Value_partial_application callee args args_arity)
  <-> value_ok (code_id_bound statics) callee
      ∧ Forall (value_ok (code_id_bound statics)) args.
Proof. apply value_ok_partial. Qed.

Lemma closures_bound_step (s : state) :
  terms_in_scope s -> closures_bound s ->
  match step s with
  | Step s' => closures_bound s'
  | Exit _ vs _ _ => Forall (value_ok (code_id_bound (state_statics s))) vs
  | Undefined => True
  end.
Proof.
  intros (P & _) V.
  apply (step_values_all (fun statics => value_ok (code_id_bound statics))
           (fun statics allocated =>
              allocated_closures_bound (code_id_bound statics) allocated
              = true)
           s closures_bound_partial V).
  - intros size tuple fields Htuple Hfields.
    destruct V as (_ & _ & Hstatics & _).
    exact (tuple_fields_ok (code_id_bound (state_statics s))
             (state_statics s) Hstatics size tuple fields Htuple Hfields).
  - intros x kind d body v hp' rd' Hexpr Hexec. rewrite Hexpr in P.
    cbn [closures_in_scope] in P. apply andb_true_iff in P as [Pd _].
    destruct V as (Hvars & _ & Hstatics & Hheap).
    exact (exec_named_ok (code_id_bound (state_statics s)) (state_statics s)
             Hstatics (state_heap s) hp' _ _ (state_vars s) d v Pd Hvars Hheap
             Hexec).
  - intros bindings body statics' Hexpr Heval. rewrite Hexpr in P.
    cbn [closures_in_scope] in P. apply andb_true_iff in P as [Pbindings _].
    pose proof (eval_static_bindings_code_bound (state_vars s) bindings
      (state_statics s) statics' Heval) as Hle.
    assert (Hle0 :
      bound_le (code_id_bound (state_statics s)) (code_id_bound statics')).
    { intros id Hid. apply Hle. unfold bound_with. rewrite Hid.
      apply orb_true_r. }
    split; [ intros v; apply value_closures_bound_le, Hle0 | ].
    destruct V as (Hvars & _ & Hstatics & _).
    apply (eval_static_bindings_ok (code_id_bound statics') (state_vars s))
      with (bindings := bindings) (statics := state_statics s).
    + exact
        (Forall_impl _
           (fun binding => value_closures_bound_le _ _ (snd binding) Hle0)
           Hvars).
    + rewrite forallb_forall in Pbindings |- *. intros binding Hin.
      exact (binding_closures_in_scope_le _ _ binding Hle
               (Pbindings binding Hin)).
    + exact
        (Forall_impl _
           (fun binding =>
              allocated_closures_bound_le _ _ (snd binding) Hle0)
           Hstatics).
    + exact Heval.
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct V as (Hvars & _).
    exact (conj (eval_simple_ok _ _ _ callee Hvars Hcallee)
             (conj (eval_simples_ok _ _ _ vs Hvars Hargs)
                (eval_simples_ok _ _ _ extra Hvars Hextra))).
  - intros ac vs _ Hargs. destruct V as (Hvars & _).
    exact (eval_simples_ok _ _ _ vs Hvars Hargs).
Qed.

Definition scope_inv (s : state) : Prop :=
  terms_in_scope s ∧ closures_bound s.

Lemma scope_inv_step (s s' : state) :
  scope_inv s -> step s = Step s' -> scope_inv s'.
Proof.
  intros [T V] Hstep. split; [ exact (terms_in_scope_step s s' T Hstep) | ].
  pose proof (closures_bound_step s T V) as Hvalues.
  rewrite Hstep in Hvalues. exact Hvalues.
Qed.

Lemma scope_inv_initial (e : expr) :
  closures_in_scope (fun _ => false) e = true ->
  scope_inv (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros He. split.
  - unfold terms_in_scope. simpl_state.
    refine (conj _ (conj I _)); [ | intros id c [=] ].
    apply (closures_in_scope_le (fun _ => false)); [ | exact He ].
    intros id [=].
  - exact
      (conj (Forall_nil _)
         (conj I (conj (Forall_nil _) (heap_values_all_empty _)))).
Qed.

Lemma scope_inv_run (n : nat) :
  ∀ s s' : state, scope_inv s -> run n s = Step s' -> scope_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (scope_inv_step s next Hinv Hstep)).
Qed.

Theorem closures_in_scope_run : closures_in_scope_run_statement.
Proof.
  intros e n s He Hrun.
  exact (proj2 (scope_inv_run n _ s (scope_inv_initial e He) Hrun)).
Qed.

(* ======================================================================== *)
(* Nothing to take out                                                      *)
(* ======================================================================== *)

(* An expression that binds nothing static has no binding of code. *)
Fixpoint without_code_nothing (e : expr) {struct e} :
  binds_statics e = false -> without_code e = e ∧ code_bindings e = []
with handler_without_code_nothing (h : continuation_handler) {struct h} :
  handler_binds_statics h = false ->
  handler_without_code h = h ∧ handler_code_bindings h = [].
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [binds_statics without_code code_bindings];
      try (intros _; split; reflexivity).
    + intros H. destruct (without_code_nothing body H) as [-> ->].
      split; reflexivity.
    + intros H. apply orb_false_iff in H as [Hbindings Hbody].
      destruct (without_code_nothing body Hbody) as [-> ->].
      assert (Hnone :
        filter is_code bindings = []
        ∧ filter (fun binding => negb (is_code binding)) bindings = bindings).
      { clear Hbody.
        induction bindings as [ | binding bindings IH];
          [ split; reflexivity | ].
        cbn [existsb] in Hbindings.
        apply orb_false_iff in Hbindings as [Hbinding Hrest].
        destruct (IH Hrest) as [IHcode IHrest].
        destruct binding as [id c | id | sym const]; simpl in Hbinding;
          try discriminate.
        cbn [filter is_code negb]. rewrite IHcode, IHrest.
        split; reflexivity. }
      destruct Hnone as [-> ->]. split; reflexivity.
    + intros H. apply orb_false_iff in H as [Hh Hbody].
      destruct (handler_without_code_nothing h Hh) as [-> ->].
      destruct (without_code_nothing body Hbody) as [-> ->].
      split; reflexivity.
    + intros H. apply orb_false_iff in H as [_ Hbody].
      destruct (without_code_nothing body Hbody) as [-> ->].
      split; reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_binds_statics handler_without_code handler_code_bindings].
    intros H. destruct (without_code_nothing handler H) as [-> ->].
    split; reflexivity.
Qed.

Lemma handlers_without_code_nothing
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics handlers = false ->
  ∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers ->
    handler_without_code h = h ∧ handler_code_bindings h = [].
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k' h'] handlers IH]; simpl;
    [ intros _ k h [] | ].
  intros H k h Hin. apply orb_false_iff in H as [Hh Hrest].
  destruct Hin as [Heq | Hin]; [ | exact (IH Hrest k h Hin) ].
  injection Heq as _ <-. exact (handler_without_code_nothing h' Hh).
Qed.

Fixpoint code_bindings_are_code (e : expr) {struct e} :
  forallb is_code (code_bindings e) = true
with handler_code_bindings_are_code (h : continuation_handler) {struct h} :
  forallb is_code (handler_code_bindings h) = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [code_bindings]; try reflexivity.
    + apply code_bindings_are_code.
    + rewrite forallb_app, (code_bindings_are_code body), andb_true_r.
      apply forallb_forall. intros binding Hin.
      apply filter_In in Hin. exact (proj2 Hin).
    + rewrite forallb_app, (code_bindings_are_code body),
        (handler_code_bindings_are_code h).
      reflexivity.
    + apply code_bindings_are_code.
  - destruct h as [params metadata handler]. cbn [handler_code_bindings].
    apply code_bindings_are_code.
Qed.

(* Binding code that is not bound, from a list with distinct code ids,
   binds each code id of the list to its code and nothing else. *)
Lemma bind_codes_from (codes : list static_binding) :
  ∀ statics : static_env,
    forallb is_code codes = true ->
    distinct (flat_map code_ids_of_binding codes) = true ->
    (∀ id : code_id,
      among id (flat_map code_ids_of_binding codes) = true ->
      lookup id (static_code statics) = None) ->
    ∃ bound : list (code_id * code),
      eval_static_bindings [] codes statics
      = Some (Static_env bound (static_values statics))
      ∧ (∀ (id : code_id) (c : code),
          Bind_code id c ∈ codes -> lookup id bound = Some c)
      ∧ (∀ (id : code_id) (c : code),
          lookup id (static_code statics) = Some c ->
          lookup id bound = Some c).
Proof.
  induction codes as [ | [id c | id | sym const] codes IH];
    intros statics Hcodes Hdistinct Hfresh; simpl in Hcodes, Hdistinct |- *;
    try discriminate.
  - exists (static_code statics).
    split; [ destruct statics; reflexivity | ].
    split; [ intros id c [] | auto ].
  - apply andb_true_iff in Hdistinct as [Hid Hrest].
    apply negb_true_iff in Hid.
    assert (Hnew : lookup id (static_code statics) = None).
    { apply Hfresh. cbn [flat_map code_ids_of_binding app].
      rewrite among_cons, Pos.eqb_refl. reflexivity. }
    rewrite Hnew.
    destruct
      (IH (Static_env ((id, c) :: static_code statics) (static_values statics))
         Hcodes Hrest)
      as (bound & Heval & Hbound & Hold).
    { intros id' Hid'. simpl.
      destruct (Pos.eqb_spec id' id) as [-> | _];
        [ rewrite Hid in Hid'; discriminate | ].
      apply Hfresh. cbn [flat_map code_ids_of_binding app].
      rewrite among_cons, Hid'. apply orb_true_r. }
    exists bound. split; [ exact Heval | ]. split.
    + intros id' c' [Heq | Hin]; [ | exact (Hbound id' c' Hin) ].
      injection Heq as <- <-. apply Hold. simpl. rewrite Pos.eqb_refl.
      reflexivity.
    + intros id' c' Hlookup. apply Hold. simpl.
      destruct (Pos.eqb_spec id' id) as [-> | _]; [ | exact Hlookup ].
      rewrite Hnew in Hlookup. discriminate.
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Hoisted.

(* The continuation for which the static bindings of the unit are at
   toplevel, and the code that the unit with its code first has bound. *)
Variable unit_exn : continuation.
Variable src : list (code_id * code).

(* The second state: the first with the bindings of code taken out of
   its terms, and all the code bound. *)
Fixpoint hoist_cont (c : cont_value) : cont_value :=
  match c with
  | Closure h vars conts recursive =>
    Closure (handler_without_code h) vars
      (map (fun binding => (fst binding, hoist_cont (snd binding))) conts)
      recursive
  | Outer k => Outer k
  | Extra_args c extra => Extra_args (hoist_cont c) extra
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    Over_application remaining remaining_arity return_arity
      (hoist_cont return_to) (hoist_cont exn_to)
  end.

Definition hoist_conts (conts : cont_env) : cont_env :=
  map (fun binding => (fst binding, hoist_cont (snd binding))) conts.

Definition code_first_env (statics : static_env) : static_env :=
  Static_env src (static_values statics).

Definition hoist_state (s : state) : state :=
  State (without_code (state_expr s)) (state_vars s)
    (hoist_conts (state_conts s)) (code_first_env (state_statics s))
      (state_heap s) (state_reads s).

(* The value of a defining expression does not depend on the code. *)
Lemma eval_named_hoist_statics (statics : static_env) (vars : env)
    (d : named) :
  eval_named (code_first_env statics) vars d = eval_named statics vars d.
Proof.
  destruct d; simpl; try reflexivity.
  all:
    destruct (eval_simples vars args) as [vs | ]; simpl; [ | reflexivity ];
    apply eval_prim_static_values; intros s _; reflexivity.
Qed.

(* Nor does what it does on a heap. *)
Lemma exec_named_hoist_statics (statics : static_env) (hp : heap)
    (rd : read_record)
    (vars : env) (d : named) :
  exec_named (code_first_env statics) hp rd vars d
    = exec_named statics hp rd vars d.
Proof.
  apply exec_named_same.
  - apply eval_named_hoist_statics.
  - intros p args _. split; reflexivity.
Qed.

Lemma resolve_hoist (k : continuation) (conts : cont_env) :
  resolve k (hoist_conts conts) = hoist_cont (resolve k conts).
Proof.
  unfold resolve, hoist_conts. rewrite lookup_map_snd.
  destruct (lookup k conts); reflexivity.
Qed.

Lemma with_extra_args_hoist (c : cont_value) (extra : list value) :
  hoist_cont (with_extra_args c extra) = with_extra_args (hoist_cont c) extra.
Proof. destruct extra; reflexivity. Qed.

Lemma rec_closures_hoist
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  handlers_bind_statics handlers = false ->
  rec_closures handlers vars (hoist_conts conts)
  = hoist_conts (rec_closures handlers vars conts).
Proof.
  intros Hnone. unfold rec_closures, hoist_conts. rewrite map_map.
  apply map_ext_in. intros [k h] Hin. cbn [fst snd hoist_cont].
  rewrite (proj1 (handlers_without_code_nothing handlers Hnone k h Hin)).
  reflexivity.
Qed.

(* What the terms of the first state bind of code is what the second
   state has bound. *)
Definition codes_known (bindings : list static_binding) : Prop :=
  ∀ (id : code_id) (c : code),
    Bind_code id c ∈ bindings -> lookup id src = Some c.

Fixpoint cont_known (c : cont_value) : Prop :=
  match c with
  | Outer _ => True
  | Closure h _ conts recursive =>
    codes_known (handler_code_bindings h)
    ∧ handlers_bind_statics recursive = false
    ∧ fold_right (fun kc acc => cont_known (snd kc) ∧ acc) True conts
  | Extra_args c _ => cont_known c
  | Over_application _ _ _ return_to exn_to =>
    cont_known return_to ∧ cont_known exn_to
  end.

Definition conts_known (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_known (snd kc) ∧ acc) True conts.

Lemma conts_known_app (conts conts' : cont_env) :
  conts_known conts -> conts_known conts' -> conts_known (conts ++ conts').
Proof.
  unfold conts_known.
  induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_known (k : continuation) (conts : cont_env) :
  conts_known conts -> cont_known (resolve k conts).
Proof.
  unfold resolve, conts_known.
  induction conts as [ | [k' c] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma with_extra_args_known (c : cont_value) (extra : list value) :
  cont_known c -> cont_known (with_extra_args c extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

Lemma rec_closures_known (vars : env) (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics all = false ->
  conts_known conts ->
  (∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers -> (k, h) ∈ all) ->
  conts_known (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hall Hconts. unfold conts_known.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros Hin. split; [ | apply IH; intros k' h' H; apply Hin; right; exact H ].
  split; [ | split; [ exact Hall | exact Hconts ] ].
  rewrite (proj2 (handlers_without_code_nothing all Hall k h
                    (Hin k h (or_introl eq_refl)))).
  intros id c [].
Qed.

(* What may follow a unit binds no code: it is the same in the second
   state, and nothing is asked of it. *)
Lemma observers_hoist (slots : list value_slot) (conts : cont_env) :
  observers slots conts -> hoist_conts conts = conts ∧ conts_known conts.
Proof.
  unfold observers, hoist_conts, conts_known.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; split; [ reflexivity | exact I ] | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  destruct (IH Hrest) as [-> Hknown].
  assert (Hthis : hoist_cont c = c ∧ cont_known c).
  { revert Hc.
    destruct c
      as [ [params metadata body] vars kept recursive | k0 | c0 extra
         | remaining remaining_arity return_arity return_to exn_to ];
      simpl; try contradiction;
      [ | intros _; split; [ reflexivity | exact I ] ].
    destruct vars; [ | contradiction ].
    destruct kept; [ | contradiction ].
    destruct recursive; [ | contradiction ].
    intros (Hno & _ & _).
    destruct
      (statics_nowhere 1%positive symbols_of_binding body
         (no_static_nowhere 1%positive body Hno))
      as (_ & Hbinds & _).
    destruct (without_code_nothing body Hbinds) as [Hsame Hnone].
    simpl. rewrite Hsame, Hnone.
    split; [ reflexivity | ].
    split; [ intros id c1 [] | ]. split; [ reflexivity | exact I ]. }
  destruct Hthis as [-> Hc_known].
  split; [ reflexivity | exact (conj Hc_known Hknown) ].
Qed.

Definition known_inv (s : state) : Prop :=
  codes_known (code_bindings (state_expr s))
  ∧ conts_known (state_conts s)
  ∧ ∀ (id : code_id) (c : code),
      lookup id (static_code (state_statics s)) = Some c ->
      lookup id src = Some c.

Definition wf_inv : state -> Prop :=
  inv unit_exn code_ids_of_binding code_id_bound.

(* The code that is bound binds nothing static. *)
Lemma wf_inv_code (s : state) (id : code_id) (c : code) :
  wf_inv s ->
  lookup id (static_code (state_statics s)) = Some c ->
  binds_statics (code_body c) = false.
Proof.
  intros (_ & _ & _ & _ & _ & S) Hlookup. pose proof (S id c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body]. simpl in Hc |- *.
  exact (proj1 (proj2 (statics_nowhere unit_exn code_ids_of_binding body Hc))).
Qed.

Lemma known_inv_over (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_known
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_known return_to ∧ cont_known exn_to.
Proof. reflexivity. Qed.

Lemma known_inv_code (statics : static_env) (hp : heap) (rd : read_record) :
  (∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    lookup id src = Some c ∧ binds_statics (code_body c) = false) ->
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    cont_known return_to ->
    cont_known exn_to ->
    known_inv
      (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).
Proof.
  intros S called c callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold known_inv. simpl_state.
  refine (conj _ (conj (conj Hreturn (conj Hexn I)) _)).
  - rewrite (proj2 (without_code_nothing _ (proj2 (S called c Hlookup)))).
    intros id c' [].
  - intros id c' H. exact (proj1 (S id c' H)).
Qed.

Lemma known_inv_handler (statics : static_env) (hp : heap) (rd : read_record) :
  (∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c -> lookup id src = Some c) ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env),
    cont_known
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    known_inv
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S params metadata handler def_vars def_conts recursive handler_vars
    Cr.
  unfold known_inv. simpl_state. simpl in Cr.
  destruct Cr as (Hh & Hrec & Cdef).
  refine (conj Hh (conj (conts_known_app _ _ _ Cdef) S)).
  exact (rec_closures_known def_vars def_conts recursive recursive Hrec Cdef
           (fun k h H => H)).
Qed.

Lemma known_inv_step (s s' : state) :
  wf_inv s -> known_inv s -> step s = Step s' -> known_inv s'.
Proof.
  intros W (K & C & S) Hstep.
  assert (Scode :
    ∀ (id : code_id) (c : code),
      lookup id (static_code (state_statics s)) = Some c ->
      lookup id src = Some c ∧ binds_statics (code_body c) = false)
    by (intros id c H; exact (conj (S id c H) (wf_inv_code s id c W H))).
  clear W. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep; cbn [code_bindings] in K.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. exact (conj K (conj C S)).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold known_inv. simpl_state.
    refine (conj _ (conj C _)).
    + intros id c Hin. apply K, in_or_app. right. exact Hin.
    + intros id c Hlookup.
      destruct (eval_static_bindings_code_in vars bindings statics statics' id
                  c Heval Hlookup)
        as [Hold | Hin]; [ exact (S id c Hold) | ].
      apply K, in_or_app. left. apply filter_In. split; [ exact Hin | ].
      reflexivity.
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold known_inv. simpl_state.
    refine (conj _ (conj (conj _ C) S)).
    + intros id c Hin. apply K, in_or_app. left. exact Hin.
    + simpl. refine (conj _ (conj eq_refl C)).
      intros id c Hin. apply K, in_or_app. right. exact Hin.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ discriminate | ].
    injection Hstep as <-. unfold known_inv. simpl_state.
    refine (conj K (conj (conts_known_app _ _ _ C) S)).
    exact (rec_closures_known vars conts handlers handlers Hbinds C
             (fun k h H => H)).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands statics _ rd_ cont_known known_inv
         (fun c extra H => H)
         known_inv_over (known_inv_code statics _ rd_ Scode)
         (known_inv_handler statics _ rd_ S) _ _ _ _ _ _ _ s'
         (resolve_known _ conts C) _ Hstep).
    apply with_extra_args_known, resolve_known, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands statics _ rd_ cont_known known_inv
         (fun c extra H => H) known_inv_over
           (known_inv_code statics _ rd_ Scode)
         (known_inv_handler statics _ rd_ S) _ vs s' (resolve_known _ conts C)
         Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    injection Hstep as <-. unfold known_inv. simpl_state.
    exact (conj (fun id c (H : Bind_code id c ∈ []) => match H with end)
             (conj C S)).
  - discriminate.
Qed.

Definition hoist_inv (s : state) : Prop :=
  wf_inv s ∧ scope_inv s ∧ known_inv s.

Lemma hoist_inv_step (s s' : state) :
  hoist_inv s -> step s = Step s' -> hoist_inv s'.
Proof.
  intros (W & P & K) Hstep.
  exact (conj (inv_step unit_exn code_ids_of_binding code_id_bound
                 code_bound_after s s' W Hstep)
           (conj (scope_inv_step s s' P Hstep)
              (known_inv_step s s' W K Hstep))).
Qed.

(* What the second run does: the same step, to the second state of the
   result, the same exit, or nothing where the first does nothing. *)
Definition result_hoisted (result result' : step_result) : Prop :=
  match result with
  | Step t => result' = Step (hoist_state t)
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Undefined => result' = Undefined
  end.

(* A group binds its symbols in the second run as in the first, where
   the first does not stop on the code of the group. *)
Lemma eval_static_bindings_hoist (vars : env)
    (bindings : list static_binding) :
  ∀ statics : static_env,
    rebinds code_ids_of_binding (code_id_bound statics) bindings = false ->
    eval_static_bindings vars
      (filter (fun binding => negb (is_code binding)) bindings)
      (code_first_env statics)
    = option_map code_first_env (eval_static_bindings vars bindings statics).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics Hrebinds; cbn [filter is_code negb eval_static_bindings].
  - reflexivity.
  - unfold rebinds in Hrebinds. simpl in Hrebinds.
    apply orb_false_iff in Hrebinds as [Hdistinct Hbound].
    apply negb_false_iff, andb_true_iff in Hdistinct as [Hfresh Hdistinct].
    apply negb_true_iff in Hfresh.
    apply orb_false_iff in Hbound as [Hid Hrest].
    unfold code_id_bound in Hid.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    rewrite <-
      (IH (Static_env ((id, c) :: static_code statics)
             (static_values statics)));
      [ reflexivity | ].
    unfold rebinds. rewrite Hdistinct. cbn [negb orb].
    apply not_true_is_false. intros Hexists.
    apply existsb_exists in Hexists as [n [Hin Hn]].
    unfold code_id_bound in Hn. simpl in Hn.
    destruct (Pos.eqb_spec n id) as [-> | _].
    + apply among_In in Hin. rewrite Hfresh in Hin. discriminate.
    + assert (Hold :
        existsb (code_id_bound statics)
          (flat_map code_ids_of_binding bindings)
        = true)
        by (apply existsb_exists; exists n; split; assumption).
      exact (diff_true_false (eq_trans (eq_sym Hold) Hrest)).
  - exact (IH statics Hrebinds).
  - cbn [code_first_env static_values static_code].
    destruct (lookup sym (static_values statics)); [ reflexivity | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | reflexivity ].
    exact
      (IH (Static_env (static_code statics)
             ((sym, allocated) :: static_values statics))
         Hrebinds).
Qed.

(* A full call. *)
Lemma call_code_hoist (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  result_hoisted
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity (hoist_cont return_to)
       (hoist_cont exn_to) (code_first_env statics) hp rd).
Proof.
  unfold result_hoisted, call_code.
  destruct (binds_statics (code_body c)) eqn:Hbinds; [ reflexivity | ].
  match goal with
  | |- context [negb ?agrees] => destruct (negb agrees)
  end;
    [ reflexivity | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)]);
    [ | reflexivity ].
  unfold hoist_state. simpl_state.
  rewrite (proj1 (without_code_nothing _ Hbinds)). reflexivity.
Qed.

Section Calls.

Variable statics : static_env.
Variable hp : heap.
Variable rd : read_record.

Hypothesis statics_bound :
  Forall
    (fun binding =>
       allocated_closures_bound (code_id_bound statics) (snd binding) = true)
    (static_values statics).

Hypothesis statics_known :
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c -> lookup id src = Some c.

Abbreviation ok := (value_ok (code_id_bound statics)).

(* The code of a closure of code that is bound is the same in the two
   static environments. *)
Lemma codes_agree_hoist (closure : value) (called : code_id)
    (slots : list (value_slot * value)) :
  ok closure ->
  closure_of statics closure = Some (called, slots) ->
  codes_agree eq statics (code_first_env statics) called.
Proof.
  intros Hclosure Hcalled.
  destruct (closure_of_ok (code_id_bound statics) statics statics_bound
              closure called slots Hclosure Hcalled)
    as [Hbound _].
  unfold codes_agree. unfold code_id_bound in Hbound.
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hlookup;
    [ | discriminate ].
  cbn [code_first_env static_code]. rewrite (statics_known called c Hlookup).
  auto.
Qed.

Lemma shape_of_call_hoist (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) :
  ok callee ->
  shapes_related eq
    (shape_of_call statics callee kind vs args_arity return_arity)
    (shape_of_call (code_first_env statics) callee kind vs args_arity
       return_arity).
Proof.
  intros Hcallee. apply shape_of_call_related; [ reflexivity | | ].
  - intros called slots.
    exact (codes_agree_hoist (called_closure callee) called slots
             (called_closure_ok _ callee Hcallee)).
  - intros tuple rest p _. apply eval_prim_static_values.
    intros sym _. reflexivity.
Qed.

(* A call of a given shape, given what the two runs do with a partial
   application. *)
Lemma run_call_hoist (return_partial return_partial' : value -> step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) :
  shapes_related eq shape shape' ->
  shape_values_all ok shape ->
  (∀ partial : value,
    ok partial ->
    result_hoisted (return_partial partial) (return_partial' partial)) ->
  result_hoisted
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape' return_arity (hoist_cont return_to)
       (hoist_cont exn_to) (code_first_env statics) hp rd).
Proof.
  intros Hshapes Hvalues Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity <-
       | closure c c' vs remaining args_arity remaining_arity <-
       | partial];
    simpl in Hvalues |- *.
  - reflexivity.
  - apply call_code_hoist.
  - exact (call_code_hoist closure c vs args_arity (code_result_arity c)
             (Over_application remaining remaining_arity return_arity
                return_to exn_to)
             exn_to statics hp rd_).
  - exact (Hpartial partial Hvalues).
Qed.

Lemma call_function_hoist_from (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) :
  ok callee ->
  Forall ok vs ->
  (∀ partial : value,
    ok partial ->
    result_hoisted (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value (hoist_cont return_to) [partial]
         (code_first_env statics) hp rd)) ->
  result_hoisted
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity
       (hoist_cont return_to) (hoist_cont exn_to) (code_first_env statics) hp
         rd).
Proof.
  intros Hcallee Hvs Hpartial. unfold call_function.
  apply run_call_hoist.
  - apply shape_of_call_hoist, Hcallee.
  - apply (shape_of_call_values_all _
             (value_ok_partial (code_id_bound statics)));
      [ | assumption | assumption ].
    intros size tuple fields.
    exact (tuple_fields_ok (code_id_bound statics) statics statics_bound
             size tuple fields).
  - exact Hpartial.
Qed.

Lemma apply_cont_value_hoist (c : cont_value) :
  ∀ vs : list value,
    cont_known c ->
    cont_values_all ok c ->
    Forall ok vs ->
    result_hoisted (apply_cont_value c vs statics hp rd)
      (apply_cont_value (hoist_cont c) vs (code_first_env statics) hp rd).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs Hknown Hvalues Hvs.
  - unfold result_hoisted. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | reflexivity ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | reflexivity ].
    simpl in Hknown. destruct Hknown as (_ & Hrec & _).
    unfold hoist_state. simpl_state. f_equal. f_equal.
    fold (hoist_conts def_conts).
    rewrite (rec_closures_hoist recursive def_vars def_conts Hrec).
    unfold hoist_conts. rewrite map_app. reflexivity.
  - reflexivity.
  - simpl in Hknown, Hvalues |- *. destruct Hvalues as [Hextra Hc].
    apply IH; [ exact Hknown | exact Hc | ].
    apply Forall_app. split; assumption.
  - simpl hoist_cont. rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ reflexivity | | reflexivity ].
    simpl in Hknown, Hvalues. destruct Hknown as [Kreturn _].
    destruct Hvalues as (Hremaining & Vreturn & _).
    apply call_function_hoist_from;
      [ exact (Forall_inv Hvs) | exact Hremaining | ].
    intros partial Hpartial.
    apply IHreturn; [ exact Kreturn | exact Vreturn | ].
    apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

Lemma call_function_hoist (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) :
  ok callee ->
  Forall ok vs ->
  cont_known return_to ->
  cont_values_all ok return_to ->
  result_hoisted
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity
       (hoist_cont return_to) (hoist_cont exn_to) (code_first_env statics) hp
         rd).
Proof.
  intros Hcallee Hvs Kreturn Vreturn.
  apply call_function_hoist_from; [ exact Hcallee | exact Hvs | ].
  intros partial Hpartial.
  apply apply_cont_value_hoist; [ exact Kreturn | exact Vreturn | ].
  apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

End Calls.

(* A step of a state of the first run is a step of the second state, to
   the second state of the result. *)
Lemma hoist_step_components (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  hoist_inv (State e vars conts statics hp rd) ->
  result_hoisted (step (State e vars conts statics hp rd))
    (step
       (State (without_code e) vars (hoist_conts conts)
          (code_first_env statics) hp rd)).
Proof.
  intros
    (W & ((_ & _ & _) & (Vvars & Vconts & Vstatics & _)) & (_ & K & S)).
  simpl_state.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [without_code]; try rewrite !step_let_exec; autorewrite with step.
  - (* Let *)
    rewrite exec_named_hoist_statics.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | reflexivity ].
    destruct (kind_eqb (kind_of_value v) kind); reflexivity.
  - (* Let_static *)
    rewrite (eval_static_bindings_hoist vars bindings statics
               (inv_no_rebinding unit_exn code_ids_of_binding code_id_bound
                  bindings body vars conts statics hp rd_ W)).
    destruct (eval_static_bindings vars bindings statics); reflexivity.
  - (* Let_cont_non_recursive *)
    reflexivity.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ reflexivity | ].
    unfold result_hoisted, hoist_state. simpl_state.
    rewrite (rec_closures_hoist handlers vars conts Hbinds).
    unfold hoist_conts. rewrite map_app. reflexivity.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | reflexivity ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | reflexivity ].
    rewrite !resolve_hoist, <- with_extra_args_hoist.
    apply (call_function_hoist statics hp rd Vstatics S).
    + exact (eval_simple_ok _ vars _ callee Vvars Hcallee).
    + exact (eval_simples_ok _ vars _ vs Vvars Hargs).
    + apply resolve_known, K.
    + apply resolve_values_all, Vconts.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    unfold apply_continuation. rewrite resolve_hoist.
    apply (apply_cont_value_hoist statics hp rd Vstatics S).
    + apply resolve_known, K.
    + apply resolve_values_all, Vconts.
    + exact (eval_simples_ok _ vars _ vs Vvars Hargs).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | reflexivity ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | reflexivity ].
    destruct (find_arm i arms) as [ac | ]; reflexivity.
  - reflexivity.
Qed.

Definition hoisted_related (s s' : state) : Prop :=
  s' = hoist_state s ∧ hoist_inv s.

Lemma hoisted_related_step (s s' : state) :
  hoisted_related s s' ->
  result_related hoisted_related (step s) (step s').
Proof.
  intros [-> Hinv].
  pose proof
    (hoist_step_components (state_expr s) (state_vars s) (state_conts s)
       (state_statics s) (state_heap s) (state_reads s))
    as Hstep.
  rewrite <- (state_eta s) in Hstep. specialize (Hstep Hinv).
  pose proof (hoist_inv_step s) as Hkept.
  unfold hoist_state at 1. unfold result_hoisted in Hstep.
  destruct (step s) as [t | k vs | ]; rewrite Hstep; simpl.
  - split; [ reflexivity | exact (Hkept t Hinv eq_refl) ].
  - auto.
  - exact I.
Qed.

End Hoisted.

(* A unit that passes and the unit with its code first behave alike,
   each run from a static environment that holds no code and in which
   what the symbols are bound to holds no closure, and from a heap whose
   cells hold no closure. *)
Lemma hoist_code_equivalent_under (slots : list value_slot) (e e' : expr)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  hoist_code e = Some e' ->
  static_code statics = [] ->
  Forall
    (fun binding =>
       allocated_closures_bound (code_id_bound statics) (snd binding) = true)
    (static_values statics) ->
  heap_values_all
    (fun v => value_closures_bound (code_id_bound statics) v = true) hp ->
  observers slots conts ->
  equivalent (State e [] conts statics hp rd) (State e' [] conts statics hp rd).
Proof.
  unfold hoist_code. cbv zeta.
  destruct (statics_at_toplevel_of_unit e) eqn:Htoplevel; simpl;
    [ | discriminate ].
  destruct (distinct (static_names code_ids_of_binding e)) eqn:Hnames; simpl;
    [ | discriminate ].
  destruct (distinct (flat_map code_ids_of_binding (code_bindings e)))
    eqn:Hcodes; simpl; [ | discriminate ].
  destruct (closures_in_scope (fun _ => false) e) eqn:Hscope;
    [ | discriminate ].
  intros [= <-] Hno_code Hallocated Hheap Hobservers.
  unfold statics_at_toplevel_of_unit in Htoplevel.
  apply existsb_exists in Htoplevel as [unit_exn [_ Htoplevel]].
  assert (Hfresh : ∀ id : code_id, lookup id (static_code statics) = None).
  { intros id. rewrite Hno_code. reflexivity. }
  destruct (bind_codes_from (code_bindings e) statics
              (code_bindings_are_code e) Hcodes (fun id _ => Hfresh id))
    as (src & Hsrc & Hknown & _).
  destruct (observers_hoist src slots conts Hobservers)
    as [Hhoist Hconts_known].
  apply
    (equivalent_trans _
       (State (without_code e) [] conts
          (Static_env src (static_values statics)) hp rd)).
  - apply (bisimulation_equivalent (hoisted_related unit_exn src)
             (hoisted_related_step unit_exn src)).
    split.
    { unfold hoist_state, code_first_env. simpl_state. rewrite Hhoist.
      reflexivity. }
    split.
    { apply (inv_under _ _ _ slots);
        [ | exact Hno_code | exact Hobservers | exact Htoplevel
        | exact Hnames ].
      intros n _. unfold code_id_bound. rewrite Hfresh. reflexivity. }
    split.
    { split.
      - unfold terms_in_scope. simpl_state.
        refine (conj _ (conj (observers_in_scope _ slots conts Hobservers) _)).
        + apply (closures_in_scope_le (fun _ => false)); [ | exact Hscope ].
          intros id [=].
        + intros id c Hlookup. rewrite Hfresh in Hlookup.
          discriminate Hlookup.
      - split; [ apply Forall_nil | ].
        split; [ exact (observers_values_all _ slots conts Hobservers) | ].
        split; [ exact Hallocated | exact Hheap ]. }
    unfold known_inv. simpl_state.
    refine (conj Hknown (conj Hconts_known _)).
    intros id c Hlookup. rewrite Hfresh in Hlookup. discriminate Hlookup.
  - apply equivalent_sym, step_equivalent.
    rewrite step_let_static, Hsrc. reflexivity.
Qed.

Lemma hoist_code_equivalent_from (e e' : expr) (statics : static_env)
    (hp : heap) (rd : read_record) :
  hoist_code e = Some e' ->
  static_code statics = [] ->
  Forall
    (fun binding =>
       allocated_closures_bound (code_id_bound statics) (snd binding) = true)
    (static_values statics) ->
  heap_values_all
    (fun v => value_closures_bound (code_id_bound statics) v = true) hp ->
  equivalent (State e [] [] statics hp rd) (State e' [] [] statics hp rd).
Proof.
  intros Hhoist Hno_code Hallocated Hheap.
  exact (hoist_code_equivalent_under [] e e' [] statics hp rd Hhoist Hno_code
           Hallocated Hheap (Forall_nil _)).
Qed.

(* The static environment of the imports of a unit is such: an import
   is no closure. *)
Lemma imports_closures_bound (bound : code_id -> bool)
    (imports : list (symbol * import_class)) :
  Forall
    (fun binding => allocated_closures_bound bound (snd binding) = true)
    (static_values (imports_statics imports)).
Proof.
  unfold imports_statics. cbn [static_values].
  induction imports as [ | [s c] imports IH]; cbn [map]; constructor;
    [ reflexivity | exact IH ].
Qed.

Theorem hoist_code_program_equivalent :
  hoist_code_program_equivalent_statement.
Proof.
  intros e e' Hhoist.
  exact
    (hoist_code_equivalent_from e e' empty_static_env empty_heap no_reads Hhoist
           eq_refl (Forall_nil _) (heap_values_all_empty _)).
Qed.

Theorem hoist_code_equivalent_imports :
  hoist_code_equivalent_imports_statement.
Proof.
  intros imports e e' Hhoist.
  exact
    (hoist_code_equivalent_from e e' (imports_statics imports) empty_heap
       no_reads
           Hhoist eq_refl (imports_closures_bound _ imports)
           (heap_values_all_empty _)).
Qed.

Theorem hoist_code_equivalent_observed :
  hoist_code_equivalent_observed_statement.
Proof.
  intros imports conts e e' Hhoist Hobservers.
  exact (hoist_code_equivalent_under [] e e' conts (imports_statics imports)
           empty_heap no_reads Hhoist eq_refl (imports_closures_bound _ imports)
           (heap_values_all_empty _) Hobservers).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The laws are proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v). Those that another file names keep
   their names and their statements: the three laws of the pass, which
   FlambdaValidatorEquivarianceProofs.v rewrites with, and
   [code_ids_of_bindings_renamed]. *)

(* By hand: the law has a premise, and the tactic states no law with
   one. FlambdaHoistStaticsProofs.v uses it. *)
Lemma filter_renamed (r : renaming) (keep : static_binding -> bool)
    (bindings : list static_binding) :
  (∀ binding : static_binding,
    keep (rename_static_binding r binding) = keep binding) ->
  filter keep (map (rename_static_binding r) bindings)
  = map (rename_static_binding r) (filter keep bindings).
Proof.
  intros Hkeep. induction bindings as [ | binding bindings IH]; simpl;
    [ reflexivity | ].
  rewrite Hkeep, IH. destruct (keep binding); reflexivity.
Qed.

Lemma code_bindings_renamed (r : renaming) (e : expr) :
  code_bindings (rename_expr r e)
  = map (rename_static_binding r) (code_bindings e).
Proof. equivariance. Qed.

Theorem code_bindings_equivariant : code_bindings_equivariant_statement.
Proof. exact code_bindings_renamed. Qed.

Lemma without_code_renamed (r : renaming) (e : expr) :
  rename_expr r (without_code e) = without_code (rename_expr r e).
Proof. equivariance. Qed.

Theorem without_code_equivariant : without_code_equivariant_statement.
Proof. exact without_code_renamed. Qed.

Lemma code_ids_of_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map code_ids_of_binding (map (rename_static_binding r) bindings)
  = flat_map code_ids_of_binding bindings.
Proof. equivariance. Qed.

(* The same law in the form that computation gives it, for the tactic to
   rewrite with: the code ids that are bound are an argument of the
   test, and a function, so the tactic meets them under a binder. *)
Lemma code_ids_of_renamed_bindings (r : renaming)
    (bindings : list static_binding) :
  flat_map
    (fun binding => code_ids_of_binding (rename_static_binding r binding))
    bindings
  = flat_map code_ids_of_binding bindings.
Proof. equivariance. Qed.
#[local] Hint Rewrite code_ids_of_renamed_bindings : equivariance.

Lemma closures_in_scope_renamed (r : renaming) (bound : code_id -> bool)
    (e : expr) :
  closures_in_scope bound (rename_expr r e) = closures_in_scope bound e.
Proof. equivariance. Qed.

Theorem closures_in_scope_equivariant :
  closures_in_scope_equivariant_statement.
Proof. exact closures_in_scope_renamed. Qed.
