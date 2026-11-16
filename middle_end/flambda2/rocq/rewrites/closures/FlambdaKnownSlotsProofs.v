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

(* Proofs of the statements of FlambdaKnownSlotsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Arith Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaStaticsInScopeSpec FlambdaUnusedStaticsSpec FlambdaKnownSymbolsSpec
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaContextWhereSpec
  FlambdaContextWhereProofs FlambdaImportsSpec FlambdaObserversSpec
  FlambdaUnitFromProofs FlambdaEquivarianceTactic FlambdaKnownSlotsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The value of a symbol or a constant                                      *)
(* ======================================================================== *)

Lemma same_simple_eq (a a' : simple) : same_simple a a' = true -> a = a'.
Proof.
  unfold same_simple. destruct (simple_eq_dec a a'); [ auto | discriminate ].
Qed.

Lemma same_simple_refl (a : simple) : same_simple a a = true.
Proof.
  unfold same_simple. destruct (simple_eq_dec a a); [ reflexivity | ].
  contradiction.
Qed.

(* A value that is that of the simple is what the simple evaluates to,
   under any variables. *)
Lemma value_is_eval (a : simple) (v : value) (vars : env) :
  value_is a v = true -> eval_simple vars a = Some v.
Proof.
  destruct v as [k | code slots | at_slot decls slots | s | tag fields
    | elements | contents | l | callee args args_arity];
    simpl; try discriminate.
  - intros H. apply same_simple_eq in H. subst a. reflexivity.
  - intros H. apply same_simple_eq in H. subst a. reflexivity.
Qed.

(* What a symbol or a constant evaluates to is its value. *)
Lemma eval_closed_value_is (vars : env) (a : simple) (v : value) :
  closed_simple a = true -> eval_simple vars a = Some v ->
  value_is a v = true.
Proof.
  destruct a as [x | sym | k]; simpl; [ discriminate | | ];
    intros _ [= <-]; apply same_simple_refl.
Qed.

Lemma closed_slots_lookup (known : known_slots) (slot : value_slot)
    (a : simple) :
  closed_slots known = true -> lookup slot known = Some a ->
  closed_simple a = true.
Proof.
  unfold closed_slots. intros Hclosed Hlookup.
  rewrite forallb_forall in Hclosed.
  exact (Hclosed (slot, a) (lookup_in _ _ _ Hlookup)).
Qed.

(* ======================================================================== *)
(* The check, but for a set of closures that has no function                *)
(* ======================================================================== *)

(* The [Let] of a [Closure_of_set] that has no function has no value:
   FlambdaSemanticsSpec.eval_named looks its slot up among the
   functions. The invariant of this file asks nothing of the value
   slots of such a [Let]; of every other set of closures it asks what
   [slots_as] asks. So a term that binds nothing static and names no
   code id has it, whatever is known: it makes no other set. What may
   follow a unit is such a term (FlambdaObserversSpec.observer). *)
Definition named_slots_checked (d : named) : list (value_slot * simple) :=
  match d with
  | Closure_of_set _ [] _ => []
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ (_ :: _) _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    named_slots_made d
  end.

Fixpoint slots_checked (known : known_slots) (e : expr) : bool :=
  match e with
  | Let _ _ d body =>
    value_slots_as known (named_slots_checked d) && slots_checked known body
  | Let_static bindings body =>
    forallb (binding_slots_checked known) bindings && slots_checked known body
  | Let_cont_non_recursive _ h body =>
    handler_slots_checked known h && slots_checked known body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_slots_checked known h) handlers
    && slots_checked known body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_slots_checked (known : known_slots) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => slots_checked known handler
  end
with binding_slots_checked (known : known_slots) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code _ c => code_slots_checked known c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const =>
    value_slots_as known (static_const_slots_made const)
  end
with code_slots_checked (known : known_slots) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => slots_checked known body
  end.

Lemma named_slots_checked_as (known : known_slots) (d : named) :
  value_slots_as known (named_slots_made d) = true ->
  value_slots_as known (named_slots_checked d) = true.
Proof.
  destruct d as [a | p args | code slots | slot [ | decl decls] slots
    | slot closure | project_from slot closure | move_from move_to closure];
    intros H; try exact H.
  reflexivity.
Qed.

(* Where the [Let] has a value, the two ask the same. *)
Lemma named_slots_checked_made (known : known_slots) (statics : static_env)
    (vars : env) (d : named) (v : value) :
  eval_named statics vars d = Some v ->
  value_slots_as known (named_slots_checked d) = true ->
  value_slots_as known (named_slots_made d) = true.
Proof.
  destruct d as [a | p args | code slots | slot [ | decl decls] slots
    | slot closure | project_from slot closure | move_from move_to closure];
    intros Heval H; try exact H.
  simpl in Heval. discriminate.
Qed.

(* What passes [slots_as] passes it. *)
Fixpoint slots_as_checked (known : known_slots) (e : expr) {struct e} :
  slots_as known e = true -> slots_checked known e = true
with handler_slots_as_checked (known : known_slots)
    (h : continuation_handler) {struct h} :
  handler_slots_as known h = true -> handler_slots_checked known h = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [slots_as slots_checked]; try (intros _; reflexivity).
    + intros H. apply andb_true_iff in H as [Hd Hbody].
      rewrite (named_slots_checked_as known d Hd),
        (slots_as_checked known body Hbody).
      reflexivity.
    + intros H. apply andb_true_iff in H as [Hbindings Hbody].
      rewrite (slots_as_checked known body Hbody), andb_true_r.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [forallb] in *; [ reflexivity | | | ];
        apply andb_true_iff in Hbindings as [Hb Hrest];
        rewrite (IH Hrest), andb_true_r.
      * exact (slots_as_checked known code_body Hb).
      * reflexivity.
      * exact Hb.
    + intros H. apply andb_true_iff in H as [Hh Hbody].
      rewrite (handler_slots_as_checked known h Hh),
        (slots_as_checked known body Hbody).
      reflexivity.
    + intros H. apply andb_true_iff in H as [Hhandlers Hbody].
      rewrite (slots_as_checked known body Hbody), andb_true_r.
      induction handlers as [ | [k h] handlers IH]; cbn [forallb] in *;
        [ reflexivity | ].
      apply andb_true_iff in Hhandlers as [Hh Hrest].
      rewrite (handler_slots_as_checked known h Hh), (IH Hrest).
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_slots_as handler_slots_checked]. apply slots_as_checked.
Qed.

Lemma code_slots_as_checked (known : known_slots) (c : code) :
  code_slots_as known c = true -> code_slots_checked known c = true.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [code_slots_as code_slots_checked]. apply slots_as_checked.
Qed.

(* What may follow a unit passes it, whatever is known: a term with no
   static binding in it, where no code id is in scope, makes no set of
   closures but one that has no function. *)
Fixpoint no_static_slots_checked (known : known_slots) (e : expr) {struct e} :
  ∀ syms : list symbol,
    binds_no_static e = true ->
    expr_statics_in_scope syms [] e = true ->
    slots_checked known e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms Hno Hscope; cbn [slots_checked]; try reflexivity.
  - cbn [binds_no_static] in Hno. cbn [expr_statics_in_scope] in Hscope.
    apply andb_true_iff in Hscope as [Hd Hbody].
    rewrite (no_static_slots_checked known body syms Hno Hbody), andb_true_r.
    destruct d as [a | p args | code slots | slot [ | decl decls] slots
      | slot closure | project_from slot closure | move_from move_to closure];
      try reflexivity; discriminate Hd.
  - discriminate Hno.
  - destruct h as [params metadata handler].
    cbn [binds_no_static handler_binds_no_static] in Hno.
    apply andb_true_iff in Hno as [Hh Hb].
    cbn [expr_statics_in_scope handler_statics_in_scope] in Hscope.
    rewrite !(no_static_names _ body Hb) in Hscope. cbn [app] in Hscope.
    apply andb_true_iff in Hscope as [Sh Sb].
    cbn [handler_slots_checked].
    rewrite (no_static_slots_checked known handler syms Hh Sh),
      (no_static_slots_checked known body syms Hb Sb).
    reflexivity.
  - cbn [binds_no_static] in Hno. apply andb_true_iff in Hno as [Hhs Hb].
    cbn [expr_statics_in_scope] in Hscope.
    rewrite !(no_static_names _ body Hb) in Hscope. cbn [app] in Hscope.
    apply andb_true_iff in Hscope as [Shs Sb].
    rewrite (no_static_slots_checked known body syms Hb Sb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [forallb] in *; [ reflexivity | ].
    apply andb_true_iff in Hhs as [Hh Hrest].
    apply andb_true_iff in Shs as [Sh Srest].
    cbn [handler_binds_no_static] in Hh.
    cbn [handler_statics_in_scope] in Sh.
    cbn [handler_slots_checked].
    rewrite (no_static_slots_checked known handler syms Hh Sh).
    exact (IH Hrest Srest).
Qed.

Lemma observers_slots_checked (known : known_slots)
    (slots : list value_slot) (conts : cont_env) :
  observers slots conts ->
  conts_terms_ok (handler_slots_checked known) conts.
Proof.
  unfold observers, conts_terms_ok.
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
  split; [ exact (no_static_slots_checked known body [] Hno Hscope) | ].
  split; [ reflexivity | exact I ].
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Section Slots_known.

Variable known : known_slots.
Hypothesis Hclosed : closed_slots known = true.

Let Pv (v : value) : Prop := value_slots_known known v = true.
Let Qv (allocated : statically_allocated_value) : Prop :=
  allocated_slots_known known allocated = true.

(* The terms. *)
Lemma slots_checked_terms_inv_step (s s' : state) :
  terms_inv (slots_checked known) (handler_slots_checked known)
    (code_slots_checked known) s ->
  step s = Step s' ->
  terms_inv (slots_checked known) (handler_slots_checked known)
    (code_slots_checked known) s'.
Proof.
  apply terms_inv_step.
  - intros x kind d body H. simpl in H. apply andb_true_iff in H as [_ H].
    exact H.
  - intros bindings body H. simpl in H.
    apply andb_true_iff in H as [Hbindings Hbody].
    split; [ exact Hbody | ]. intros id code Hin.
    rewrite forallb_forall in Hbindings. exact (Hbindings _ Hin).
  - intros k h body H. simpl in H. apply andb_true_iff in H. exact H.
  - intros handlers body H. simpl in H. apply andb_true_iff in H. exact H.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

(* The values. *)
Lemma slot_values_known_in (slot_values : list (value_slot * value)) :
  forallb (fun '(_, contents) => value_slots_known known contents)
    slot_values
  = true ->
  ∀ (w : value_slot) (contents : value),
    (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros H w contents Hin. rewrite forallb_forall in H.
  exact (H (w, contents) Hin).
Qed.

Lemma statics_slots_known_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  lookup s (static_values statics) = Some allocated ->
  Qv allocated.
Proof.
  intros Hstatics Hs. rewrite Forall_forall in Hstatics.
  exact (Hstatics (s, allocated) (lookup_in _ _ _ Hs)).
Qed.

Lemma eval_simple_slots_known (vars : env) (s : simple) (v : value) :
  env_values_all Pv vars -> eval_simple vars s = Some v -> Pv v.
Proof.
  intros Hvars. destruct s as [x | sym | k]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_slots_known (vars : env) :
  env_values_all Pv vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    forallb (value_slots_known known) vs = true.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_slots_known vars s v Hvars Hs).
    exact (IH rest eq_refl).
Qed.

(* The value slots of a set that passes the check, evaluated. *)
Lemma eval_value_slots_slots_known (vars : env) :
  env_values_all Pv vars ->
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    value_slots_as known value_slots = true ->
    eval_value_slots vars value_slots = Some slot_values ->
    slot_values_as known slot_values = true
    ∧ forallb (fun '(_, contents) => value_slots_known known contents)
        slot_values
      = true.
Proof.
  intros Hvars.
  induction value_slots as [ | [slot s] value_slots IH]; intros slot_values;
    simpl.
  - intros _ [= <-]. split; reflexivity.
  - intros Has. apply andb_true_iff in Has as [Hslot Hrest].
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    destruct (IH rest Hrest eq_refl) as [Hrest_as Hrest_known].
    rewrite Hrest_as, Hrest_known,
      (eval_simple_slots_known vars s v Hvars Hs), !andb_true_r.
    split; [ | reflexivity ].
    destruct (lookup slot known) as [a | ] eqn:Hlookup; [ | reflexivity ].
    apply same_simple_eq in Hslot. subst a.
    exact (eval_closed_value_is vars s v
             (closed_slots_lookup known slot s Hclosed Hlookup) Hs).
Qed.

Lemma values_slots_known_Forall (vs : list value) :
  forallb (value_slots_known known) vs = true -> Forall Pv vs.
Proof.
  intros H. rewrite forallb_forall in H. apply Forall_forall. exact H.
Qed.

(* A closure of which the invariant holds, read as a closure of a set:
   the values of its value slots are as [known] says, and the invariant
   holds of each. *)
Lemma closure_in_set_slots_known (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  Pv v ->
  closure_in_set statics v = Some (slot, decls, slot_values) ->
  slot_values_as known slot_values = true
  ∧ ∀ (w : value_slot) (contents : value),
      (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros Hstatics Hv. unfold closure_in_set.
  destruct v as [k | code slots | at_slot set_decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Has Hslots].
    intros [= <- <- <-].
    exact (conj Has (slot_values_known_in slots Hslots)).
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Has Hslots].
    destruct (lookup at_slot set_decls); [ | discriminate ].
    simpl. intros [= <- <- <-].
    exact (conj Has (slot_values_known_in slots Hslots)).
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + pose proof (statics_slots_known_lookup statics s _ Hstatics Hs)
        as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Has Hslots].
      intros [= <- <- <-].
      exact (conj Has (slot_values_known_in slots Hslots)).
    + pose proof (statics_slots_known_lookup statics s _ Hstatics Hs)
        as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Has Hslots].
      destruct (lookup at_slot set_decls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      exact (conj Has (slot_values_known_in slots Hslots)).
Qed.

Lemma value_slots_known_of_numbers : holds_of_numbers Pv.
Proof. intros k. split; reflexivity. Qed.

Lemma value_slots_known_of_blocks : holds_of_blocks Pv.
Proof.
  intros tag fields.
  exact (forallb_forall (value_slots_known known) fields).
Qed.

Lemma value_slots_known_of_arrays : holds_of_arrays Pv.
Proof.
  intros elements.
  exact (forallb_forall (value_slots_known known) elements).
Qed.

Lemma value_slots_known_in_static_blocks (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_blocks statics Pv.
Proof.
  intros Hstatics s tag fields _ Hs v Hin.
  pose proof (statics_slots_known_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma value_slots_known_in_static_arrays (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_arrays statics Pv.
Proof.
  intros Hstatics s elements _ Hs v Hin.
  pose proof (statics_slots_known_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma project_function_slot_slots_known (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  Pv v -> project_function_slot statics move_from move_to v = Some w -> Pv w.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v as [k | code slots | at_slot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros Hproject.
    apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
    exact Hv.
  - destruct (Pos.eqb at_slot move_from); [ | discriminate ].
    destruct (lookup at_slot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros Hproject.
      apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
      reflexivity.
    + destruct (Pos.eqb at_slot move_from); [ | discriminate ].
      destruct (lookup at_slot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      destruct (lookup move_to syms); [ | discriminate ].
      simpl. intros [= <-]. reflexivity.
Qed.

Lemma eval_named_slots_known (statics : static_env) (vars : env) (d : named)
    (v : value) :
  value_slots_as known (named_slots_made d) = true ->
  env_values_all Pv vars ->
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  eval_named statics vars d = Some v ->
  Pv v.
Proof.
  intros Hd Hvars Hstatics.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    simpl.
  - exact (eval_simple_slots_known vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl. intros Hprim.
    apply (eval_prim_preserves Pv statics p vs v value_slots_known_of_numbers
             value_slots_known_of_blocks value_slots_known_of_arrays
             (value_slots_known_in_static_blocks statics Hstatics)
             (value_slots_known_in_static_arrays statics Hstatics));
      [ | exact Hprim ].
    pose proof (eval_simples_slots_known vars Hvars args vs Hargs) as Hvs.
    rewrite forallb_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl. simpl in Hd.
    destruct (eval_value_slots_slots_known vars Hvars slots vs Hd Hslots)
      as [Has Hknown].
    rewrite Has, Hknown. reflexivity.
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl. simpl in Hd.
    destruct (eval_value_slots_slots_known vars Hvars slots vs Hd Hslots)
      as [Has Hknown].
    rewrite Has, Hknown. reflexivity.
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    destruct (closure_of statics k) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics k code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_slots_known statics k at_slot decls slot_values
                Hstatics (eval_simple_slots_known vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics k) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      [ | discriminate ].
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_slots_known statics k at_slot decls slot_values
                Hstatics (eval_simple_slots_known vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    exact (project_function_slot_slots_known statics move_from move_to k v
             (eval_simple_slots_known vars closure k Hvars Hk)).
Qed.

Lemma eval_static_const_slots_known (const : static_const) (vars : env)
    (allocated : statically_allocated_value) :
  value_slots_as known (static_const_slots_made const) = true ->
  env_values_all Pv vars ->
  eval_static_const vars const = Some allocated ->
  Qv allocated.
Proof.
  intros Hconst Hvars.
  destruct const as [code slots | slot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl. simpl in Hconst.
    destruct (eval_value_slots_slots_known vars Hvars slots vs Hconst Hslots)
      as [Has Hknown].
    rewrite Has, Hknown. reflexivity.
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl. simpl in Hconst.
    destruct (eval_value_slots_slots_known vars Hvars slots vs Hconst Hslots)
      as [Has Hknown].
    rewrite Has, Hknown. reflexivity.
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_slots_known vars Hvars fields vs Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_slots_known vars contents v Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_slots_known vars Hvars elements vs Helements).
Qed.

Lemma eval_static_bindings_slots_known (vars : env) :
  env_values_all Pv vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb (binding_slots_checked known) bindings = true ->
    Forall (fun binding => Qv (snd binding)) (static_values statics) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    Forall (fun binding => Qv (snd binding)) (static_values statics').
Proof.
  intros Hvars.
  induction bindings as [ | [id code | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hvalid [= <-]. exact Hvalid.
  - intros H Hvalid. apply andb_true_iff in H as [_ Hrest].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). exact Hvalid.
  - intros Hrest Hvalid. exact (IH _ _ Hrest Hvalid).
  - intros H Hvalid. apply andb_true_iff in H as [Hconst Hrest].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Heval;
      simpl; [ | discriminate ].
    apply (IH _ _ Hrest). simpl. apply Forall_cons; [ | exact Hvalid ].
    exact (eval_static_const_slots_known const vars allocated Hconst Hvars
             Heval).
Qed.

Lemma value_slots_known_partial (callee : value) (args : list value)
    (args_arity : arity) :
  Pv (Value_partial_application callee args args_arity)
  <-> Pv callee ∧ Forall Pv args.
Proof.
  unfold Pv. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity.
Qed.

Lemma state_slots_known_step (s : state) :
  slots_checked known (state_expr s) = true ->
  state_slots_known known s ->
  match step s with
  | Step s' => state_slots_known known s'
  | Exit _ vs _ _ => Forall Pv vs
  | Undefined => True
  end.
Proof.
  intros P Hvalid.
  apply (step_values_all
           (fun _ v => value_slots_known known v = true)
           (fun _ allocated => allocated_slots_known known allocated = true)
           s (fun _ => value_slots_known_partial) Hvalid).
  - intros size tuple fields Htuple Hfields.
    destruct Hvalid as (_ & _ & Hstatics & _).
    apply (tuple_fields_forall _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros field v Hprim.
    refine (eval_prim_preserves Pv (state_statics s) _ [tuple] v
              value_slots_known_of_numbers value_slots_known_of_blocks
              value_slots_known_of_arrays
              (value_slots_known_in_static_blocks (state_statics s) Hstatics)
              (value_slots_known_in_static_arrays (state_statics s) Hstatics)
              _ Hprim).
    intros w [<- | []]. exact Htuple.
  - intros x kind d body v hp' rd' Hexpr Hexec. rewrite Hexpr in P.
    cbn [slots_checked] in P. apply andb_true_iff in P as [Pd _].
    destruct Hvalid as (Hvars & _ & Hstatics & Hheap).
    refine
      (exec_named_values_all Pv (state_statics s) (state_heap s) hp'
         (state_reads s) rd'
              (state_vars s) d v value_slots_known_of_numbers
              value_slots_known_of_blocks value_slots_known_of_arrays
              (value_slots_known_in_static_blocks (state_statics s) Hstatics)
              (value_slots_known_in_static_arrays (state_statics s) Hstatics)
              (fun _ => eq_refl) Hheap _ _ Hexec).
    + intros p args vs _ Hvs.
      exact (proj1 (forallb_forall _ _)
               (eval_simples_slots_known (state_vars s) Hvars args vs Hvs)).
    + intros _ w Heval.
      exact (eval_named_slots_known (state_statics s) (state_vars s) d w
               (named_slots_checked_made known _ _ d w Heval Pd) Hvars
               Hstatics Heval).
  - intros bindings body statics' Hexpr Heval.
    split; [ exact (fun v H => H) | ].
    rewrite Hexpr in P. simpl in P. apply andb_true_iff in P as [Pbindings _].
    destruct Hvalid as (Hvars & _ & Hstatics & _).
    exact (eval_static_bindings_slots_known (state_vars s) Hvars bindings
             (state_statics s) statics' Pbindings Hstatics Heval).
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct Hvalid as (Hvars & _).
    split; [ | split ].
    + exact (eval_simple_slots_known (state_vars s) (ap_callee ap) callee
               Hvars Hcallee).
    + exact (values_slots_known_Forall vs
               (eval_simples_slots_known (state_vars s) Hvars (ap_args ap) vs
                  Hargs)).
    + exact (values_slots_known_Forall extra
               (eval_simples_slots_known (state_vars s) Hvars
                  (exn_extra_simples ap) extra Hextra)).
  - intros ac vs _ Hargs. destruct Hvalid as (Hvars & _).
    exact (values_slots_known_Forall vs
             (eval_simples_slots_known (state_vars s) Hvars (ac_args ac) vs
                Hargs)).
Qed.

(* The two together: what the refinement goes by. *)
Definition slots_inv (s : state) : Prop :=
  terms_inv (slots_checked known) (handler_slots_checked known)
    (code_slots_checked known) s
  ∧ state_slots_known known s.

Lemma slots_inv_kept : kept_by_step slots_inv.
Proof.
  intros s s' [Hterms Hvalues] Hstep.
  split; [ exact (slots_checked_terms_inv_step s s' Hterms Hstep) | ].
  pose proof (state_slots_known_step s (proj1 Hterms) Hvalues) as H.
  rewrite Hstep in H. exact H.
Qed.

Lemma slots_inv_initial (e : expr) :
  slots_as known e = true
    -> slots_inv (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros He. split.
  - exact (terms_inv_initial _ _ _ e (slots_as_checked known e He)).
  - exact
      (conj (Forall_nil _)
         (conj I (conj (Forall_nil _) (heap_values_all_empty _)))).
Qed.

(* The same from a static environment that binds something already. *)
Lemma slots_inv_from (e : expr) (statics : static_env) :
  slots_as known e = true ->
  Forall (fun binding => code_slots_as known (snd binding) = true)
    (static_code statics) ->
  Forall (fun binding => allocated_slots_known known (snd binding) = true)
    (static_values statics) ->
  slots_inv (State e [] [] statics empty_heap no_reads).
Proof.
  intros He Hcode Hvalues. split.
  - unfold terms_inv. simpl_state.
    refine (conj (slots_as_checked known e He) (conj I _)).
    intros id c Hlookup. rewrite Forall_forall in Hcode.
    exact (code_slots_as_checked known c
             (Hcode (id, c) (lookup_in id c _ Hlookup))).
  - exact
      (conj (Forall_nil _)
         (conj I (conj Hvalues (heap_values_all_empty _)))).
Qed.

(* The same with what may follow the unit in the continuation
   environment. *)
Lemma slots_inv_observed (slots : list value_slot) (e : expr)
    (conts : cont_env) (statics : static_env) :
  slots_as known e = true ->
  Forall (fun binding => code_slots_as known (snd binding) = true)
    (static_code statics) ->
  Forall (fun binding => allocated_slots_known known (snd binding) = true)
    (static_values statics) ->
  observers slots conts ->
  slots_inv (State e [] conts statics empty_heap no_reads).
Proof.
  intros He Hcode Hvalues Hobservers. split.
  - unfold terms_inv. simpl_state.
    refine
      (conj (slots_as_checked known e He)
         (conj (observers_slots_checked known slots conts Hobservers) _)).
    intros id c Hlookup. rewrite Forall_forall in Hcode.
    exact (code_slots_as_checked known c
             (Hcode (id, c) (lookup_in id c _ Hlookup))).
  - exact (conj (Forall_nil _)
             (conj (observers_values_all _ slots conts Hobservers)
                (conj Hvalues (heap_values_all_empty _)))).
Qed.

Lemma slots_inv_run (n : nat) :
  ∀ s s' : state, slots_inv s -> run n s = Step s' -> slots_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (slots_inv_kept s next Hinv Hstep)).
Qed.

End Slots_known.

Theorem slots_as_kept : slots_as_kept_statement.
Proof.
  intros known e n s Hclosed He Hrun.
  exact (proj2 (slots_inv_run known Hclosed n _ s (slots_inv_initial known e He)
                  Hrun)).
Qed.

(* ======================================================================== *)
(* Reading a slot                                                           *)
(* ======================================================================== *)

(* The value of a slot that is listed, in a closure of which the
   invariant holds. *)
Lemma slot_values_as_lookup (known : known_slots)
    (slot_values : list (value_slot * value)) (slot : value_slot)
    (a : simple) (v : value) :
  slot_values_as known slot_values = true ->
  lookup slot known = Some a ->
  lookup slot slot_values = Some v ->
  value_is a v = true.
Proof.
  unfold slot_values_as. intros Has Hknown Hlookup.
  rewrite forallb_forall in Has.
  pose proof (Has (slot, v) (lookup_in _ _ _ Hlookup)) as H. simpl in H.
  rewrite Hknown in H. exact H.
Qed.

Theorem read_slot_sound : read_slot_sound_statement.
Proof.
  intros known syms s d v (Hvars & _ & Hstatics & _).
  set (statics := state_statics s) in *. set (vars := state_vars s) in *.
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_slot]; try (intros H; exact H).
  - destruct (lookup slot known) as [a | ] eqn:Hknown;
      [ | intros H; exact H ].
    destruct (simple_statics_in_scope syms a); [ | intros H; exact H ].
    simpl.
    destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    destruct (closure_of statics k) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics k code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_slots_known known statics k at_slot decls
                slot_values Hstatics
                (eval_simple_slots_known known vars closure k Hvars Hk)
                Hin_set)
      as [Has _].
    exact (value_is_eval a v vars
             (slot_values_as_lookup known slot_values slot a v Has Hknown
                Hlookup)).
  - destruct (lookup slot known) as [a | ] eqn:Hknown;
      [ | intros H; exact H ].
    destruct (simple_statics_in_scope syms a); [ | intros H; exact H ].
    simpl.
    destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics k) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      [ | discriminate ].
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_slots_known known statics k at_slot decls
                slot_values Hstatics
                (eval_simple_slots_known known vars closure k Hvars Hk)
                Hin_set)
      as [Has _].
    exact (value_is_eval a v vars
             (slot_values_as_lookup known slot_values slot a v Has Hknown
                Hlookup)).
Qed.

(* The same where the defining expression is run on a heap: a primitive
   is left as it is, and what is not a primitive has its value by
   [eval_named], as has what it is rewritten to. *)
Lemma read_slot_sound_exec (known : known_slots) (syms : list symbol)
    (s : state) (hp hp' : heap) (rd rd' : read_record) (d : named)
    (v : value) :
  state_slots_known known s ->
  exec_named (state_statics s) hp rd (state_vars s) d = Some (v, hp', rd') ->
  exec_named (state_statics s) hp rd (state_vars s) (read_slot known syms d)
  = Some (v, hp', rd').
Proof.
  intros Hknown Hd.
  assert (Hnot :
    ∀ (p : prim) (args : list simple),
      d <> Prim p args -> read_slot known syms d <> Prim p args).
  { intros p args.
    destruct d as [a | q qargs | code slots | slot decls slots | slot closure
      | project_from slot closure | move_from move_to closure];
      cbn [read_slot]; try (intros H; exact H);
      (destruct (lookup slot known) as [a | ]; [ | intros H; exact H ]);
      (destruct (simple_statics_in_scope syms a); [ | intros H; exact H ]);
      intros _; discriminate. }
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    try (pose proof (fun H => exec_named_pure _ _ _ _ _ _ _ _ H Hd) as Hpure;
         destruct (Hpure eq_refl) as (Hw & -> & ->);
         rewrite exec_named_not_prim
           by (intros ? ?; apply Hnot; discriminate);
         rewrite (read_slot_sound known syms s _ v Hknown Hw);
         reflexivity).
  exact Hd.
Qed.

(* ======================================================================== *)
(* What the rewrite keeps                                                   *)
(* ======================================================================== *)

Lemma read_slot_slots_made (known : known_slots) (syms : list symbol)
    (d : named) :
  named_slots_made (read_slot known syms d) = named_slots_made d.
Proof.
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_slot]; try reflexivity;
    (destruct (lookup slot known) as [a | ]; [ | reflexivity ]);
    destruct (simple_statics_in_scope syms a); reflexivity.
Qed.

(* The rewrite makes no set of closures and changes none. *)
Fixpoint read_slots_slots_as_expr (known known' : known_slots) (e : expr)
    {struct e} :
  ∀ syms : list symbol,
    slots_as known' (read_slots known syms e) = slots_as known' e
with read_slots_slots_as_handler (known known' : known_slots)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    handler_slots_as known' (read_slots_in_handler known syms h)
    = handler_slots_as known' h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms; cbn [read_slots slots_as]; try reflexivity.
    + rewrite read_slot_slots_made,
        (read_slots_slots_as_expr known known' body).
      reflexivity.
    + rewrite (read_slots_slots_as_expr known known' body). f_equal.
      generalize (flat_map symbols_of_binding bindings ++ syms).
      intros all.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map forallb]; [ reflexivity | | | ].
      * cbn [read_slots_in_binding code_is_stub].
        destruct (cm_is_stub metadata); [ rewrite IH; reflexivity | ].
        cbn [read_slots_in_code binding_slots_as code_slots_as].
        rewrite (read_slots_slots_as_expr known known' code_body), IH.
        reflexivity.
      * cbn [read_slots_in_binding]. rewrite IH. reflexivity.
      * cbn [read_slots_in_binding]. rewrite IH. reflexivity.
    + rewrite (read_slots_slots_as_handler known known' h),
        (read_slots_slots_as_expr known known' body).
      reflexivity.
    + rewrite (read_slots_slots_as_expr known known' body). f_equal.
      generalize (static_names symbols_of_binding body ++ syms).
      intros all.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (read_slots_slots_as_handler known known' h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros syms.
    cbn [read_slots_in_handler handler_slots_as].
    apply read_slots_slots_as_expr.
Qed.

Theorem read_slots_slots_as : read_slots_slots_as_statement.
Proof.
  intros known known' syms e. apply read_slots_slots_as_expr.
Qed.

Lemma read_slots_bindings_slots_as (known known' : known_slots)
    (syms : list symbol) (bindings : list static_binding) :
  forallb (binding_slots_as known')
    (map (read_slots_in_binding known syms) bindings)
  = forallb (binding_slots_as known') bindings.
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings IH];
    cbn [map forallb]; [ reflexivity | | | ].
  - cbn [read_slots_in_binding code_is_stub].
    destruct (cm_is_stub metadata); [ rewrite IH; reflexivity | ].
    cbn [read_slots_in_code binding_slots_as code_slots_as].
    rewrite read_slots_slots_as_expr, IH. reflexivity.
  - cbn [read_slots_in_binding]. rewrite IH. reflexivity.
  - cbn [read_slots_in_binding]. rewrite IH. reflexivity.
Qed.

Lemma read_slot_slots_checked (known : known_slots) (syms : list symbol)
    (d : named) :
  named_slots_checked (read_slot known syms d) = named_slots_checked d.
Proof.
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_slot]; try reflexivity;
    (destruct (lookup slot known) as [a | ]; [ | reflexivity ]);
    destruct (simple_statics_in_scope syms a); reflexivity.
Qed.

(* The same of the check that the invariant has. *)
Fixpoint read_slots_slots_checked_expr (known known' : known_slots) (e : expr)
    {struct e} :
  ∀ syms : list symbol,
    slots_checked known' (read_slots known syms e) = slots_checked known' e
with read_slots_slots_checked_handler (known known' : known_slots)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    handler_slots_checked known' (read_slots_in_handler known syms h)
    = handler_slots_checked known' h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms; cbn [read_slots slots_checked]; try reflexivity.
    + rewrite read_slot_slots_checked,
        (read_slots_slots_checked_expr known known' body).
      reflexivity.
    + rewrite (read_slots_slots_checked_expr known known' body). f_equal.
      generalize (flat_map symbols_of_binding bindings ++ syms).
      intros all.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map forallb]; [ reflexivity | | | ].
      * cbn [read_slots_in_binding code_is_stub].
        destruct (cm_is_stub metadata); [ rewrite IH; reflexivity | ].
        cbn [read_slots_in_code binding_slots_checked code_slots_checked].
        rewrite (read_slots_slots_checked_expr known known' code_body), IH.
        reflexivity.
      * cbn [read_slots_in_binding]. rewrite IH. reflexivity.
      * cbn [read_slots_in_binding]. rewrite IH. reflexivity.
    + rewrite (read_slots_slots_checked_handler known known' h),
        (read_slots_slots_checked_expr known known' body).
      reflexivity.
    + rewrite (read_slots_slots_checked_expr known known' body). f_equal.
      generalize (static_names symbols_of_binding body ++ syms).
      intros all.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (read_slots_slots_checked_handler known known' h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros syms.
    cbn [read_slots_in_handler handler_slots_checked].
    apply read_slots_slots_checked_expr.
Qed.

Lemma read_slots_bindings_slots_checked (known known' : known_slots)
    (syms : list symbol) (bindings : list static_binding) :
  forallb (binding_slots_checked known')
    (map (read_slots_in_binding known syms) bindings)
  = forallb (binding_slots_checked known') bindings.
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings IH];
    cbn [map forallb]; [ reflexivity | | | ].
  - cbn [read_slots_in_binding code_is_stub].
    destruct (cm_is_stub metadata); [ rewrite IH; reflexivity | ].
    cbn [read_slots_in_code binding_slots_checked code_slots_checked].
    rewrite read_slots_slots_checked_expr, IH. reflexivity.
  - cbn [read_slots_in_binding]. rewrite IH. reflexivity.
  - cbn [read_slots_in_binding]. rewrite IH. reflexivity.
Qed.

(* The rewrite binds what the term binds. *)
Fixpoint read_slots_binds_statics (known : known_slots) (e : expr) :
  ∀ syms : list symbol,
    binds_statics (read_slots known syms e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms; cbn [read_slots binds_statics]; try reflexivity.
  - apply read_slots_binds_statics.
  - rewrite (read_slots_binds_statics known body). f_equal.
    generalize (flat_map symbols_of_binding bindings ++ syms). intros all.
    induction bindings as [ | [id c | id | sym const] bindings IH];
      cbn [map existsb read_slots_in_binding];
      try destruct (code_is_stub c); cbn [binding_binds]; congruence.
  - destruct h as [params metadata handler].
    cbn [read_slots_in_handler handler_binds_statics].
    rewrite !read_slots_binds_statics. reflexivity.
  - rewrite (read_slots_binds_statics known body). f_equal.
    generalize (static_names symbols_of_binding body ++ syms). intros all.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [map existsb read_slots_in_handler handler_binds_statics];
      [ reflexivity | ].
    rewrite read_slots_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* Refinement                                                               *)
(* ======================================================================== *)

Section Refines.

Variable known : known_slots.
Hypothesis Hclosed : closed_slots known = true.

Let inv : state -> Prop := slots_inv known.

Let Hkept : kept_by_step inv := slots_inv_kept known Hclosed.

(* The invariant does not look at the expression of a state but to ask
   the check of it. *)
Lemma slots_inv_expr (e e' : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  slots_checked known e' = slots_checked known e ->
  inv (State e vars conts statics hp rd)
    <-> inv (State e' vars conts statics hp rd).
Proof.
  intros Has.
  unfold inv, slots_inv, terms_inv, state_slots_known, state_values_all.
  simpl_state. rewrite Has. reflexivity.
Qed.

Lemma slots_inv_let_static_code (done after : list static_binding)
    (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body : expr) (syms : list symbol) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  inv
    (State
       (Let_static
          (done
           ++ Bind_code id (Code params my_closure ret exn metadata code_body)
           :: after)
          body)
       vars conts statics hp rd) ->
  inv
    (State
       (Let_static
          (done
           ++ Bind_code id
                (Code params my_closure ret exn metadata
                   (read_slots known syms code_body))
           :: after)
          body)
       vars conts statics hp rd).
Proof.
  apply slots_inv_expr. cbn [slots_checked]. rewrite !forallb_app.
  cbn [forallb binding_slots_checked code_slots_checked].
  rewrite read_slots_slots_checked_expr. reflexivity.
Qed.

Lemma slots_inv_let_cont_recursive_handler
    (done after : list (continuation * continuation_handler))
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler : expr) (syms : list symbol) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  inv
    (State
       (Let_cont_recursive
          (done ++ (k, Handler params metadata handler) :: after) body)
       vars conts statics hp rd) ->
  inv
    (State
       (Let_cont_recursive
          (done
           ++ (k, Handler params metadata (read_slots known syms handler))
           :: after)
          body)
       vars conts statics hp rd).
Proof.
  apply slots_inv_expr. cbn [slots_checked]. rewrite !forallb_app.
  cbn [forallb handler_slots_checked].
  rewrite read_slots_slots_checked_expr. reflexivity.
Qed.

(* Down the term. The code of a group and the handlers of a recursive
   [Let_cont] are rewritten one at a time
   (FlambdaContextWhereSpec.let_static_code_refines_from_statement and
   the like), each from the states that have the invariant. *)
Fixpoint read_slots_refines_aux (e : expr) {struct e} :
  ∀ (syms : list symbol) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    inv (State e vars conts statics hp rd) ->
    refines (State e vars conts statics hp rd)
      (State (read_slots known syms e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms vars conts statics hp rd Hinv; cbn [read_slots];
    try apply refines_refl.
  - (* Let *)
    apply let_refines_step_forward.
    + intros v hp' rd' Hd.
      exact (read_slot_sound_exec known syms
               (State (Let x kind d body) vars conts statics hp rd) hp hp' rd
               rd' d v (proj2 Hinv) Hd).
    + intros v hp' rd' Hd Hkind. apply read_slots_refines_aux.
      apply (Hkept _ _ Hinv).
      rewrite step_let_exec, Hd, Hkind, kind_eqb_refl. reflexivity.
  - (* Let_static *)
    generalize (flat_map symbols_of_binding bindings ++ syms). intros all.
    apply (refines_trans _
      (State
         (Let_static (map (read_slots_in_binding known all) bindings) body)
         vars conts statics hp rd)).
    + assert (Hcode :
        ∀ done : list static_binding,
          inv
            (State (Let_static (done ++ bindings) body) vars conts statics hp
               rd)
          ->
          refines
            (State (Let_static (done ++ bindings) body) vars conts statics hp
               rd)
            (State
               (Let_static
                  (done ++ map (read_slots_in_binding known all) bindings)
                  body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | | | ].
        - cbn [read_slots_in_binding code_is_stub].
          destruct (cm_is_stub metadata).
          { (* The code of a stub is left. *)
            specialize
              (IH
                 (done
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata
                           code_body)])).
            rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone). }
          cbn [read_slots_in_code].
          pose proof
            (slots_inv_let_static_code done bindings id params my_closure ret
               exn metadata code_body all body vars conts statics _ rd_ Hdone)
            as Hdone'.
          apply (refines_trans _
            (State
               (Let_static
                  (done
                   ++ Bind_code id
                        (Code params my_closure ret exn metadata
                           (read_slots known all code_body))
                   :: bindings)
                  body)
               vars conts statics hp rd)).
          + apply (let_static_code_refines_from inv done id params my_closure
                     ret exn metadata code_body
                     (read_slots known all code_body) bindings body vars
                     conts statics hp rd Hkept Hdone Hdone').
            * symmetry. apply read_slots_binds_statics.
            * intros vs callee callee_vars return_to exn_to statics' hp' rd' _ _
              _.
              split.
              -- apply slots_inv_expr, read_slots_slots_checked_expr.
              -- intros Hentry. apply read_slots_refines_aux. exact Hentry.
          + specialize
              (IH
                 (done
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata
                           (read_slots known all code_body))])).
            rewrite <- !app_assoc in IH. cbn [app] in IH.
            exact (IH Hdone').
        - cbn [read_slots_in_binding].
          specialize (IH (done ++ [Bind_deleted_code id])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone).
        - cbn [read_slots_in_binding].
          specialize (IH (done ++ [Bind_symbol sym const])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone). }
      exact (Hcode [] Hinv).
    + assert (Hinv' :
        inv
          (State
             (Let_static (map (read_slots_in_binding known all) bindings)
                body)
             vars conts statics hp rd)).
      { revert Hinv. apply slots_inv_expr. cbn [slots_checked].
        rewrite read_slots_bindings_slots_checked. reflexivity. }
      apply let_static_refines_step.
      destruct
        (eval_static_bindings vars
           (map (read_slots_in_binding known all) bindings) statics)
        as [extended | ] eqn:Heval;
        [ | exact I ].
      apply read_slots_refines_aux. apply (Hkept _ _ Hinv').
      rewrite step_let_static, Heval. reflexivity.
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler]. cbn [read_slots_in_handler].
    generalize (static_names symbols_of_binding body ++ syms). intros all.
    assert (Hinv' :
      inv
        (State
           (Let_cont_non_recursive k
              (Handler params metadata (read_slots known all handler)) body)
           vars conts statics hp rd)).
    { revert Hinv. apply slots_inv_expr.
      cbn [slots_checked handler_slots_checked].
      rewrite read_slots_slots_checked_expr. reflexivity. }
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (read_slots known all handler)) body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_from inv k params metadata handler
               (read_slots known all handler) body vars conts statics hp rd
               Hkept Hinv Hinv').
      intros vs handler_vars conts' statics' hp' rd' _ _. split.
      * apply slots_inv_expr, read_slots_slots_checked_expr.
      * intros Hentry. apply read_slots_refines_aux. exact Hentry.
    + apply let_cont_non_recursive_refines_step.
      apply read_slots_refines_aux. apply (Hkept _ _ Hinv').
      apply step_let_cont_non_recursive.
  - (* Let_cont_recursive *)
    generalize (static_names symbols_of_binding body ++ syms). intros all.
    set (read :=
           fun kh : continuation * continuation_handler =>
             let '(k, h) := kh in (k, read_slots_in_handler known all h)).
    apply (refines_trans _
      (State (Let_cont_recursive (map read handlers) body) vars conts
         statics hp rd)).
    + assert (Hhandlers :
        ∀ done : list (continuation * continuation_handler),
          inv
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp rd)
          ->
          refines
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp rd)
            (State (Let_cont_recursive (done ++ map read handlers) body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | ].
        unfold read at 1. cbn [read_slots_in_handler].
        pose proof
          (slots_inv_let_cont_recursive_handler done handlers k params
             metadata handler all body vars conts statics _ rd_ Hdone)
          as Hdone'.
        apply (refines_trans _
          (State
             (Let_cont_recursive
                (done
                 ++
                   (k, Handler params metadata (read_slots known all handler))
                 :: handlers)
                body)
             vars conts statics hp rd)).
        - apply (let_cont_recursive_handler_refines_from inv done k params
                   metadata handler (read_slots known all handler) handlers
                   body vars conts statics _ rd_ Hkept Hdone Hdone').
          + symmetry. apply read_slots_binds_statics.
          + intros vs handler_vars conts' statics' hp' rd' _ _. split.
            * apply slots_inv_expr, read_slots_slots_checked_expr.
            * intros Hentry. apply read_slots_refines_aux. exact Hentry.
        - specialize
            (IH
               (done
                ++ [(k,
                     Handler params metadata
                       (read_slots known all handler))])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone'). }
      exact (Hhandlers [] Hinv).
    + assert (Hinv' :
        inv
          (State (Let_cont_recursive (map read handlers) body) vars conts
             statics hp rd)).
      { revert Hinv. apply slots_inv_expr. cbn [slots_checked]. f_equal.
        clear. induction handlers as [ | [k h] handlers IH];
          cbn [map forallb]; [ reflexivity | ].
        unfold read at 1. rewrite read_slots_slots_checked_handler, IH.
        reflexivity. }
      apply let_cont_recursive_refines_step; [ reflexivity | ].
      intros Hbinds. apply read_slots_refines_aux. apply (Hkept _ _ Hinv').
      rewrite step_let_cont_recursive, Hbinds. reflexivity.
Qed.

End Refines.

Theorem read_slots_program_refines : read_slots_program_refines_statement.
Proof.
  intros known syms e Hclosed Has. unfold program_refines.
  apply (read_slots_refines_aux known Hclosed).
  exact (slots_inv_initial known e Has).
Qed.

Theorem read_known_slots_program_refines :
  read_known_slots_program_refines_statement.
Proof.
  intros e. unfold read_known_slots. cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e)
    eqn:Hcheck;
    [ | apply refines_refl ].
  apply andb_true_iff in Hcheck as [Hclosed Has].
  exact (read_slots_program_refines _ [] e Hclosed Has).
Qed.

Theorem read_slots_refines_from : read_slots_refines_from_statement.
Proof.
  intros known syms e statics Hclosed Has Hcode Hvalues.
  apply (read_slots_refines_aux known Hclosed).
  exact (slots_inv_from known e statics Has Hcode Hvalues).
Qed.

Theorem read_known_slots_refines_imports :
  read_known_slots_refines_imports_statement.
Proof.
  intros imports e. unfold unit_refines, read_known_slots. cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e)
    eqn:Hcheck;
    [ | apply refines_refl ].
  apply andb_true_iff in Hcheck as [Hclosed Has].
  apply (read_slots_refines_from _ [] e (imports_statics imports) Hclosed Has).
  - apply Forall_nil.
  - unfold imports_statics. cbn [static_values]. apply Forall_forall.
    intros [s allocated] Hin.
    apply in_map_iff in Hin as ([s' c] & Heq & _).
    injection Heq as _ <-. reflexivity.
Qed.

Theorem read_slots_refines_observed : read_slots_refines_observed_statement.
Proof.
  intros known syms e conts statics Hclosed Has Hcode Hvalues Hobservers.
  apply (read_slots_refines_aux known Hclosed).
  exact (slots_inv_observed known [] e conts statics Has Hcode Hvalues
           Hobservers).
Qed.

Theorem read_known_slots_refines_observed :
  read_known_slots_refines_observed_statement.
Proof.
  intros imports conts e Hobservers. unfold read_known_slots. cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e)
    eqn:Hcheck;
    [ | apply refines_refl ].
  apply andb_true_iff in Hcheck as [Hclosed Has].
  apply
    (read_slots_refines_observed _ [] e conts (imports_statics imports)
       Hclosed Has);
    [ apply Forall_nil | | exact Hobservers ].
  unfold imports_statics. cbn [static_values]. apply Forall_forall.
  intros [s allocated] Hin.
  apply in_map_iff in Hin as ([s' c] & Heq & _).
  injection Heq as _ <-. reflexivity.
Qed.

(* What the static environment of the imports binds its symbols to has
   no value slot. *)
Lemma imports_slots_known (known : known_slots)
    (imports : list (symbol * import_class)) :
  Forall (fun binding => allocated_slots_known known (snd binding) = true)
    (static_values (imports_statics imports)).
Proof.
  unfold imports_statics. cbn [static_values]. apply Forall_forall.
  intros [s allocated] Hin.
  apply in_map_iff in Hin as ([s' c] & Heq & _).
  injection Heq as _ <-. reflexivity.
Qed.

Theorem read_known_slots_anywhere_program_refines :
  read_known_slots_anywhere_program_refines_statement.
Proof.
  intros e. unfold read_known_slots_anywhere. cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e)
    eqn:Hcheck;
    [ | apply refines_refl ].
  apply andb_true_iff in Hcheck as [Hclosed Has].
  exact (read_slots_program_refines _ _ e Hclosed Has).
Qed.

Theorem read_known_slots_anywhere_refines_imports :
  read_known_slots_anywhere_refines_imports_statement.
Proof.
  intros imports e. unfold unit_refines, read_known_slots_anywhere. cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e)
    eqn:Hcheck;
    [ | apply refines_refl ].
  apply andb_true_iff in Hcheck as [Hclosed Has].
  exact (read_slots_refines_from _ _ e (imports_statics imports) Hclosed Has
           (Forall_nil _) (imports_slots_known _ imports)).
Qed.

Theorem read_known_slots_anywhere_refines_observed :
  read_known_slots_anywhere_refines_observed_statement.
Proof.
  intros imports conts e Hobservers. unfold read_known_slots_anywhere.
  cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e)
    eqn:Hcheck;
    [ | apply refines_refl ].
  apply andb_true_iff in Hcheck as [Hclosed Has].
  exact (read_slots_refines_observed _ _ e conts (imports_statics imports)
           Hclosed Has (Forall_nil _) (imports_slots_known _ imports)
           Hobservers).
Qed.

(* ======================================================================== *)
(* Row 4                                                                    *)
(* ======================================================================== *)

Lemma simple_in_scope_incl (syms syms' : list symbol) (a : simple) :
  incl syms syms' ->
  simple_statics_in_scope syms a = true ->
  simple_statics_in_scope syms' a = true.
Proof.
  intros Hincl. destruct a as [x | sym | k]; simpl; [ auto | | auto ].
  intros H. apply among_In, Hincl, among_In, H.
Qed.

Lemma read_slot_in_scope (known : known_slots) (syms syms' : list symbol)
    (ids : list code_id) (d : named) :
  incl syms syms' ->
  named_statics_in_scope syms' ids d = true ->
  named_statics_in_scope syms' ids (read_slot known syms d) = true.
Proof.
  intros Hincl Hd.
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_slot]; try exact Hd;
    (destruct (lookup slot known) as [a | ]; [ | exact Hd ]);
    (destruct (simple_statics_in_scope syms a) eqn:Ha; [ | exact Hd ]);
    cbn [named_statics_in_scope];
    exact (simple_in_scope_incl syms syms' a Hincl Ha).
Qed.

(* The rewrite changes the body of code only, so what a binding binds,
   by a function that does not look into code, is what it bound. *)
Section Names_kept.

Variable names_of : static_binding -> list positive.
Hypothesis names_of_code :
  ∀ (id : code_id) (c c' : code),
    names_of (Bind_code id c') = names_of (Bind_code id c).

Lemma read_slots_binding_names (known : known_slots) (syms : list symbol)
    (binding : static_binding) :
  names_of (read_slots_in_binding known syms binding) = names_of binding.
Proof.
  destruct binding as [id c | id | sym const]; cbn [read_slots_in_binding];
    [ destruct (code_is_stub c); [ reflexivity | apply names_of_code ]
    | reflexivity | reflexivity ].
Qed.

Lemma read_slots_bindings_names (known : known_slots) (syms : list symbol)
    (bindings : list static_binding) :
  flat_map names_of (map (read_slots_in_binding known syms) bindings)
  = flat_map names_of bindings.
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map flat_map];
    [ reflexivity | ].
  rewrite read_slots_binding_names, IH. reflexivity.
Qed.

Fixpoint read_slots_static_names (known : known_slots) (e : expr)
    {struct e} :
  ∀ syms : list symbol,
    static_names names_of (read_slots known syms e) = static_names names_of e
with read_slots_handler_static_names (known : known_slots)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    handler_static_names names_of (read_slots_in_handler known syms h)
    = handler_static_names names_of h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms; cbn [read_slots static_names]; try reflexivity.
    + apply read_slots_static_names.
    + rewrite read_slots_bindings_names,
        (read_slots_static_names known body).
      reflexivity.
    + rewrite (read_slots_handler_static_names known h),
        (read_slots_static_names known body).
      reflexivity.
    + rewrite (read_slots_static_names known body). f_equal.
      generalize (static_names symbols_of_binding body ++ syms).
      intros all.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (read_slots_handler_static_names known h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros syms.
    cbn [read_slots_in_handler handler_static_names].
    apply read_slots_static_names.
Qed.

End Names_kept.

Lemma read_slots_group_consts (known : known_slots) (syms : list symbol)
    (all : list symbol) (ids : list code_id)
    (bindings : list static_binding) :
  ∀ before : list symbol,
    group_consts_in_scope before all ids
      (map (read_slots_in_binding known syms) bindings)
    = group_consts_in_scope before all ids bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros before; cbn [map read_slots_in_binding];
    try destruct (code_is_stub c); cbn [group_consts_in_scope];
    [ reflexivity | apply IH | apply IH | apply IH | ].
  destruct (is_closure_const const); rewrite IH; reflexivity.
Qed.

Fixpoint read_slots_in_scope_expr (known : known_slots) (e : expr)
    {struct e} :
  ∀ (syms syms' : list symbol) (ids : list code_id),
    incl syms syms' ->
    expr_statics_in_scope syms' ids e = true ->
    expr_statics_in_scope syms' ids (read_slots known syms e) = true
with read_slots_in_scope_handler (known : known_slots)
    (h : continuation_handler) {struct h} :
  ∀ (syms syms' : list symbol) (ids : list code_id),
    incl syms syms' ->
    handler_statics_in_scope syms' ids h = true ->
    handler_statics_in_scope syms' ids (read_slots_in_handler known syms h)
    = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms syms' ids Hincl He; cbn [read_slots]; try exact He.
    + (* Let *)
      cbn [expr_statics_in_scope] in He |- *.
      apply andb_true_iff in He as [Hd Hbody].
      rewrite (read_slot_in_scope known syms syms' ids d Hincl Hd).
      exact (read_slots_in_scope_expr known body syms syms' ids Hincl Hbody).
    + (* Let_static *)
      assert (Hbindings :
        ∀ (all all' : list symbol) (ids' : list code_id),
          incl all all' ->
          forallb (binding_statics_in_scope all' ids') bindings = true ->
          forallb (binding_statics_in_scope all' ids')
            (map (read_slots_in_binding known all) bindings)
          = true).
      { clear He. intros all all' ids' Hall.
        induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          cbn [map forallb]; [ reflexivity | | | ]; intros Hcode;
          apply andb_true_iff in Hcode as [Hc Hrest];
          apply andb_true_iff; (split; [ | exact (IH Hrest) ]).
        - cbn [read_slots_in_binding code_is_stub].
          destruct (cm_is_stub metadata); [ exact Hc | ].
          cbn [read_slots_in_code binding_statics_in_scope
                 code_statics_in_scope] in Hc |- *.
          exact (read_slots_in_scope_expr known code_body all all' ids' Hall
                   Hc).
        - reflexivity.
        - reflexivity. }
      cbn [expr_statics_in_scope] in He |- *. cbv zeta in He |- *.
      rewrite
        (read_slots_bindings_names symbols_of_binding
           (fun _ _ _ => eq_refl)),
        (read_slots_bindings_names declared_code_ids_of_binding
           (fun _ _ _ => eq_refl)),
        (read_slots_bindings_names closure_symbols_of_binding
           (fun _ _ _ => eq_refl)),
        read_slots_group_consts.
      apply andb_true_iff in He as [He Hbody].
      apply andb_true_iff in He as [Hgroup Hcode].
      assert (Hall :
        incl (flat_map symbols_of_binding bindings ++ syms)
          (flat_map symbols_of_binding bindings ++ syms'))
        by exact (incl_app_app (incl_refl _) Hincl).
      apply andb_true_iff. split; [ apply andb_true_iff; split | ].
      * exact Hgroup.
      * exact (Hbindings _ _ _ Hall Hcode).
      * exact (read_slots_in_scope_expr known body _ _ _ Hall Hbody).
    + (* Let_cont_non_recursive *)
      cbn [expr_statics_in_scope] in He |- *.
      apply andb_true_iff in He as [Hh Hbody].
      rewrite
        (read_slots_static_names symbols_of_binding (fun _ _ _ => eq_refl)),
        (read_slots_static_names declared_code_ids_of_binding
           (fun _ _ _ => eq_refl)).
      apply andb_true_iff. split.
      * exact (read_slots_in_scope_handler known h _ _ _
                 (incl_app_app (incl_refl _) Hincl) Hh).
      * exact (read_slots_in_scope_expr known body syms syms' ids Hincl
                 Hbody).
    + (* Let_cont_recursive *)
      assert (Hrec :
        ∀ (all all' : list symbol) (ids' : list code_id),
          incl all all' ->
          forallb (fun '(_, h) => handler_statics_in_scope all' ids' h)
            handlers
          = true ->
          forallb (fun '(_, h) => handler_statics_in_scope all' ids' h)
            (map (fun '(k, h) => (k, read_slots_in_handler known all h))
               handlers)
          = true).
      { clear He. intros all all' ids' Hall.
        induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
          [ reflexivity | ]; intros Hhandlers.
        apply andb_true_iff in Hhandlers as [Hh Hrest].
        apply andb_true_iff. split; [ | exact (IH Hrest) ].
        exact (read_slots_in_scope_handler known h all all' ids' Hall Hh). }
      cbn [expr_statics_in_scope] in He |- *.
      apply andb_true_iff in He as [Hhandlers Hbody].
      rewrite
        (read_slots_static_names symbols_of_binding (fun _ _ _ => eq_refl)),
        (read_slots_static_names declared_code_ids_of_binding
           (fun _ _ _ => eq_refl)).
      apply andb_true_iff. split.
      * exact (Hrec _ _ _ (incl_app_app (incl_refl _) Hincl) Hhandlers).
      * exact (read_slots_in_scope_expr known body syms syms' ids Hincl
                 Hbody).
  - destruct h as [params metadata handler].
    intros syms syms' ids Hincl Hh.
    cbn [read_slots_in_handler handler_statics_in_scope] in Hh |- *.
    exact (read_slots_in_scope_expr known handler syms syms' ids Hincl Hh).
Qed.

Theorem read_slots_in_scope : read_slots_in_scope_statement.
Proof.
  intros known syms syms' ids e Hincl He.
  exact (read_slots_in_scope_expr known e syms syms' ids Hincl He).
Qed.

Theorem read_known_slots_in_scope : read_known_slots_in_scope_statement.
Proof.
  intros imports e He. unfold read_known_slots. cbv zeta.
  destruct
    (closed_slots (slots_known (slots_made e))
     && slots_as (slots_known (slots_made e)) e);
    [ | exact He ].
  unfold statics_in_scope in *.
  exact (read_slots_in_scope_expr _ e [] imports [] (incl_nil_l imports) He).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* Where a law below has the proof [equivariance], it is proved from
   the definitions by the tactic of FlambdaEquivarianceTactic.v. The
   laws that ask that what is known holds symbols and constants only
   are proved by hand, but for that of [slots_as], which the tactic
   proves from the law of [value_slots_as] given with its premise, and
   that of [read_slots], which comes from its law with what is known
   renamed too. *)

Lemma rename_closed_simple (r : renaming) (a : simple) :
  closed_simple a = true -> rename_simple r a = a.
Proof. destruct a as [x | sym | k]; [ discriminate | reflexivity .. ]. Qed.

Lemma closed_simple_renamed (r : renaming) (a : simple) :
  closed_simple (rename_simple r a) = closed_simple a.
Proof. equivariance. Qed.

(* A renaming is a permutation: two simples are the same once renamed
   if and only if they are. *)
Lemma same_simple_renamed (r : renaming) (a a' : simple) :
  same_simple (rename_simple r a) (rename_simple r a') = same_simple a a'.
Proof. equivariance. Qed.

(* A value slot with what it holds, renamed. *)
Definition rename_slot (r : renaming) (entry : value_slot * simple)
    : value_slot * simple :=
  (fst entry, rename_simple r (snd entry)).

Lemma named_slots_made_renamed (r : renaming) (d : named) :
  named_slots_made (rename_named r d)
  = map (rename_slot r) (named_slots_made d).
Proof. equivariance. Qed.

Lemma static_const_slots_made_renamed (r : renaming) (const : static_const) :
  static_const_slots_made (rename_static_const r const)
  = map (rename_slot r) (static_const_slots_made const).
Proof. equivariance. Qed.

Lemma slots_made_renamed (r : renaming) (e : expr) :
  slots_made (rename_expr r e) = map (rename_slot r) (slots_made e).
Proof. equivariance. Qed.

(* What is known is the same: an entry that is kept holds a symbol or a
   constant, which no renaming changes. *)
Lemma slots_known_test_renamed (r : renaming) (slot : value_slot)
    (a : simple) (made : list (value_slot * simple)) :
  forallb
    (fun '(slot', a') =>
       negb (Pos.eqb slot' slot) || same_simple a' (rename_simple r a))
    (map (rename_slot r) made)
  = forallb
      (fun '(slot', a') => negb (Pos.eqb slot' slot) || same_simple a' a)
      made.
Proof.
  induction made as [ | [slot' a'] made IH];
    cbn [map forallb rename_slot fst snd]; [ reflexivity | ].
  rewrite same_simple_renamed, IH. reflexivity.
Qed.

Lemma slots_known_filter_renamed (r : renaming)
    (all entries : list (value_slot * simple)) :
  filter
    (fun '(slot, a) =>
       closed_simple a
       && forallb
            (fun '(slot', a') => negb (Pos.eqb slot' slot) || same_simple a' a)
            (map (rename_slot r) all))
    (map (rename_slot r) entries)
  = filter
      (fun '(slot, a) =>
         closed_simple a
         && forallb
              (fun '(slot', a') =>
                 negb (Pos.eqb slot' slot) || same_simple a' a)
              all)
      entries.
Proof.
  induction entries as [ | [slot a] entries IH];
    cbn [map filter rename_slot fst snd]; [ reflexivity | ].
  rewrite closed_simple_renamed, slots_known_test_renamed, IH.
  destruct (closed_simple a) eqn:Ha; cbn [andb]; [ | reflexivity ].
  unfold rename_slot. cbn [fst snd].
  rewrite (rename_closed_simple r a Ha). reflexivity.
Qed.

Lemma slots_known_renamed (r : renaming)
    (made : list (value_slot * simple)) :
  slots_known (map (rename_slot r) made) = slots_known made.
Proof. exact (slots_known_filter_renamed r made made). Qed.

(* The check is the same of a term renamed, where what is known holds
   symbols and constants only. *)
Lemma value_slots_as_renamed (r : renaming) (known : known_slots)
    (value_slots : list (value_slot * simple)) :
  closed_slots known = true ->
  value_slots_as known (map (rename_slot r) value_slots)
  = value_slots_as known value_slots.
Proof.
  intros Hclosed. unfold value_slots_as.
  induction value_slots as [ | [slot a] value_slots IH];
    cbn [map forallb rename_slot fst snd]; [ reflexivity | ].
  rewrite IH.
  destruct (lookup slot known) as [a' | ] eqn:Hknown; [ | reflexivity ].
  replace (same_simple (rename_simple r a) a') with (same_simple a a');
    [ reflexivity | ].
  rewrite <- (same_simple_renamed r a a'),
    (rename_closed_simple r a'
       (closed_slots_lookup known slot a' Hclosed Hknown)).
  reflexivity.
Qed.

#[local] Hint Rewrite named_slots_made_renamed
  static_const_slots_made_renamed : equivariance.
#[local] Hint Rewrite value_slots_as_renamed using assumption
  : equivariance.

Lemma slots_as_renamed (r : renaming) (known : known_slots)
    (Hclosed : closed_slots known = true) (e : expr) :
  slots_as known (rename_expr r e) = slots_as known e.
Proof. equivariance. Qed.

(* A renaming renames no symbol, so the symbols in scope are the
   same. *)
Section Names_renamed.

Variable r : renaming.
Variable names_of : static_binding -> list positive.
Hypothesis names_of_renamed :
  ∀ binding : static_binding,
    names_of (rename_static_binding r binding) = names_of binding.

#[local] Hint Rewrite names_of_renamed : equivariance.

Lemma static_names_of_renamed (e : expr) :
  static_names names_of (rename_expr r e) = static_names names_of e.
Proof. equivariance. Qed.

End Names_renamed.

Lemma symbols_of_binding_renamed (r : renaming) (binding : static_binding) :
  symbols_of_binding (rename_static_binding r binding)
  = symbols_of_binding binding.
Proof. equivariance. Qed.

(* The rewrite, with what is known renamed too: no premise. The three
   laws before it are given to the tactic in the forms that
   computation gives. *)
Lemma lookup_slots_renamed (r : renaming) (slot : value_slot)
    (known : known_slots) :
  lookup slot
    (map (fun p : value_slot * simple => (fst p, rename_simple r (snd p)))
       known)
  = option_map (rename_simple r) (lookup slot known).
Proof. equivariance. Qed.

Lemma symbols_of_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map (fun a => symbols_of_binding (rename_static_binding r a)) bindings
  = flat_map symbols_of_binding bindings.
Proof. equivariance. Qed.

Lemma symbol_names_renamed (r : renaming) (e : expr) :
  static_names symbols_of_binding (rename_expr r e)
  = static_names symbols_of_binding e.
Proof. equivariance. Qed.

#[local] Hint Rewrite lookup_slots_renamed symbols_of_bindings_renamed
  symbol_names_renamed : equivariance.

Lemma read_slots_all_renamed (r : renaming) (known : known_slots)
    (syms : list symbol) (e : expr) :
  read_slots
    (map (fun p : value_slot * simple => (fst p, rename_simple r (snd p)))
       known)
    syms (rename_expr r e)
  = rename_expr r (read_slots known syms e).
Proof. equivariance. Qed.

(* What is known is as it was once renamed, where it holds symbols
   and constants only. *)
Lemma closed_slots_same (r : renaming) (known : known_slots) :
  closed_slots known = true ->
  map (fun p : value_slot * simple => (fst p, rename_simple r (snd p))) known
  = known.
Proof.
  induction known as [ | [slot a] known IH]; cbn [closed_slots forallb map];
    [ reflexivity | ].
  intros Hclosed. apply andb_true_iff in Hclosed as [Ha Hclosed].
  cbn [fst snd] in Ha |- *.
  rewrite (rename_closed_simple r a Ha), (IH Hclosed).
  reflexivity.
Qed.

Lemma read_slots_renamed (r : renaming) (known : known_slots)
    (Hclosed : closed_slots known = true) (e : expr) :
  ∀ syms : list symbol,
    read_slots known syms (rename_expr r e)
    = rename_expr r (read_slots known syms e).
Proof.
  intros syms. rewrite <- (closed_slots_same r known Hclosed) at 1.
  apply read_slots_all_renamed.
Qed.

Theorem read_known_slots_equivariant : read_known_slots_equivariant_statement.
Proof.
  intros r e. unfold read_known_slots. cbv zeta.
  rewrite slots_made_renamed, slots_known_renamed.
  destruct (closed_slots (slots_known (slots_made e))) eqn:Hclosed;
    cbn [andb]; [ | reflexivity ].
  rewrite (slots_as_renamed r _ Hclosed e).
  destruct (slots_as (slots_known (slots_made e)) e); [ | reflexivity ].
  exact (read_slots_renamed r _ Hclosed e []).
Qed.

Theorem read_known_slots_anywhere_equivariant :
  read_known_slots_anywhere_equivariant_statement.
Proof.
  intros r e. unfold read_known_slots_anywhere. cbv zeta.
  rewrite slots_made_renamed, slots_known_renamed.
  destruct (closed_slots (slots_known (slots_made e))) eqn:Hclosed;
    cbn [andb]; [ | reflexivity ].
  rewrite (slots_as_renamed r _ Hclosed e).
  destruct (slots_as (slots_known (slots_made e)) e); [ | reflexivity ].
  rewrite (static_names_of_renamed r symbols_of_binding
             (symbols_of_binding_renamed r) e).
  exact (read_slots_renamed r _ Hclosed e _).
Qed.

(* ======================================================================== *)
(* Five units                                                               *)
(* ======================================================================== *)

Theorem read_known_slots_reads : read_known_slots_reads_statement.
Proof. reflexivity. Qed.

Theorem read_known_slots_variable : read_known_slots_variable_statement.
Proof. reflexivity. Qed.

Theorem read_known_slots_anywhere_out_of_scope :
  read_known_slots_anywhere_out_of_scope_statement.
Proof. split; reflexivity. Qed.

Theorem read_known_slots_out_of_scope :
  read_known_slots_out_of_scope_statement.
Proof. split; reflexivity. Qed.

Theorem read_known_slots_in_stub : read_known_slots_in_stub_statement.
Proof. reflexivity. Qed.

Theorem read_known_slots_stub_makes : read_known_slots_stub_makes_statement.
Proof. reflexivity. Qed.
