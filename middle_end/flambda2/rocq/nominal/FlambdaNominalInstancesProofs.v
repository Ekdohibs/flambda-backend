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

(* Proofs of the statements of FlambdaNominalInstancesSpec.v. Nothing here
   needs review.

   An instance whose equality is [=] has its laws from five facts about
   its action: the identity, composition, that the action depends on the
   renaming as a function only, the free names of a renamed value, and
   that a renaming fixing the free names fixes the value
   ([variables_instance_nominal]). For closed expressions the same facts
   are those of [rename_expr], with alpha-equivalence where a renaming
   fixes the free names. A binder is shown to be an abstraction by
   opening each side with the rule of [alpha_equivalent] for that
   binder, and relating the two opened scopes under the binder. *)

From Stdlib Require Import ZArith List Bool Eqdep_dec.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaBindersProofs
  FlambdaSemanticsSpec FlambdaContextSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaFreeNamesProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaEquivarianceProofs FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs FlambdaNominalInstancesSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* An instance whose equality is [=]                                        *)
(* ======================================================================== *)

Section Variables_instance.

Context {T : Type}.

Variable rename : renaming -> T -> T.

Variable free : T -> list variable.

Hypothesis rename_id : ∀ t : T, rename identity_renaming t = t.

Hypothesis rename_compose :
  ∀ (second first : renaming) (t : T),
    rename (compose_renaming second first) t
    = rename second (rename first t).

Hypothesis rename_ext :
  ∀ r r' : renaming,
    (∀ x : variable, rename_variable r x = rename_variable r' x) ->
    ∀ t : T, rename r t = rename r' t.

Hypothesis free_renamed :
  ∀ (r : renaming) (t : T),
    free (rename r t) = map (rename_variable r) (free t).

Hypothesis rename_fixes :
  ∀ (r : renaming) (t : T),
    (∀ x : variable, among x (free t) = true -> rename_variable r x = x) ->
    rename r t = t.

Lemma variables_instance_nominal :
  nominal
    (Contains_names (fun p t => rename (variables_renaming p) t) free eq).
Proof.
  constructor; simpl.
  - reflexivity.
  - intros t t' Hequal. symmetry. exact Hequal.
  - intros t t' t'' Hequal Hequal'. congruence.
  - intros t. apply rename_id.
  - intros second first t.
    exact (rename_compose (variables_renaming second)
             (variables_renaming first) t).
  - intros p p' t Hsame. apply rename_ext. intros x. apply Hsame.
  - intros p t t' ->. reflexivity.
  - intros p t name. rewrite free_renamed.
    exact (among_map_injective (rename_variable (variables_renaming p))
             (rename_variable_eqb (variables_renaming p)) name (free t)).
  - intros t t' -> name. reflexivity.
  - intros p t Hfixes. apply rename_fixes. intros x Hx.
    exact (Hfixes x Hx).
Qed.

End Variables_instance.

(* ======================================================================== *)
(* A renaming that fixes the free variables                                 *)
(* ======================================================================== *)

Section Fixes.

Variable r : renaming.

Lemma rename_simple_fixes (s : simple) :
  (∀ x : variable,
    among x (free_variables_of_simple s) = true -> rename_variable r x = x) ->
  rename_simple r s = s.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hfixes; [ | reflexivity.. ].
  rewrite (Hfixes x); [ reflexivity | ].
  rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma rename_simples_fixes (args : list simple) :
  (∀ x : variable,
    among x (free_variables_of_simples args) = true ->
    rename_variable r x = x) ->
  map (rename_simple r) args = args.
Proof.
  unfold free_variables_of_simples.
  induction args as [ | s args IH]; cbn [map flat_map]; intros Hfixes;
    [ reflexivity | ].
  rewrite rename_simple_fixes, IH; [ reflexivity | | ]; intros x Hx;
    apply Hfixes; rewrite among_app, Hx; [ apply orb_true_r | reflexivity ].
Qed.

Lemma rename_value_slots_fixes (slots : list (value_slot * simple)) :
  (∀ x : variable,
    among x (free_variables_of_value_slots slots) = true ->
    rename_variable r x = x) ->
  map (fun slot => (fst slot, rename_simple r (snd slot))) slots = slots.
Proof.
  unfold free_variables_of_value_slots.
  induction slots as [ | [slot s] slots IH]; cbn [map flat_map fst snd];
    intros Hfixes; [ reflexivity | ].
  rewrite rename_simple_fixes, IH; [ reflexivity | | ]; intros x Hx;
    apply Hfixes; rewrite among_app, Hx; [ apply orb_true_r | reflexivity ].
Qed.

Lemma rename_named_fixes (d : named) :
  (∀ x : variable,
    among x (free_variables_of_named d) = true -> rename_variable r x = x) ->
  rename_named r d = d.
Proof.
  destruct d; cbn [rename_named free_variables_of_named]; intros Hfixes;
    rewrite ?rename_simple_fixes, ?rename_simples_fixes,
      ?rename_value_slots_fixes;
    first [ reflexivity | exact Hfixes ].
Qed.

Lemma rename_static_const_fixes (const : static_const) :
  (∀ x : variable,
    among x (free_variables_of_static_const const) = true ->
    rename_variable r x = x) ->
  rename_static_const r const = const.
Proof.
  destruct const;
    cbn [rename_static_const free_variables_of_static_const];
    intros Hfixes;
    rewrite ?rename_simple_fixes, ?rename_simples_fixes,
      ?rename_value_slots_fixes;
    first [ reflexivity | exact Hfixes ].
Qed.

Lemma map_fst_rename_env (vars : env) :
  map fst (rename_env r vars) = map (rename_variable r) (map fst vars).
Proof.
  induction vars as [ | [x v] vars IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma rename_env_fixes (vars : env) :
  (∀ x : variable,
    among x (map fst vars) = true -> rename_variable r x = x) ->
  rename_env r vars = vars.
Proof.
  induction vars as [ | [x v] vars IH]; intros Hfixes; [ reflexivity | ].
  change (rename_env r ((x, v) :: vars))
    with ((rename_variable r x, v) :: rename_env r vars).
  rewrite (Hfixes x), IH; [ reflexivity | | ].
  - intros y Hy. apply Hfixes. cbn [map fst].
    rewrite among_cons, Hy. apply orb_true_r.
  - cbn [map fst]. rewrite among_cons, Pos.eqb_refl. reflexivity.
Qed.

End Fixes.

Theorem simple_nominal : simple_nominal_statement.
Proof.
  apply variables_instance_nominal.
  - exact rename_simple_id.
  - exact rename_simple_compose.
  - exact rename_simple_ext.
  - exact free_variables_of_simple_renamed.
  - exact rename_simple_fixes.
Qed.

Theorem named_nominal : named_nominal_statement.
Proof.
  apply variables_instance_nominal.
  - exact rename_named_id.
  - exact rename_named_compose.
  - exact rename_named_ext.
  - exact free_variables_of_named_renamed.
  - exact rename_named_fixes.
Qed.

Theorem static_const_nominal : static_const_nominal_statement.
Proof.
  apply variables_instance_nominal.
  - exact rename_static_const_id.
  - exact rename_static_const_compose.
  - exact rename_static_const_ext.
  - exact free_variables_of_static_const_renamed.
  - exact rename_static_const_fixes.
Qed.

Theorem env_nominal : env_nominal_statement.
Proof.
  apply variables_instance_nominal.
  - exact rename_env_id.
  - exact rename_env_compose.
  - exact rename_env_ext.
  - exact map_fst_rename_env.
  - exact rename_env_fixes.
Qed.

(* ======================================================================== *)
(* Closed expressions                                                       *)
(* ======================================================================== *)

Theorem rename_closed_expr_term : rename_closed_expr_term_statement.
Proof.
  intros r [e Hclosed]. unfold rename_closed_expr. simpl.
  destruct (bool_dec (code_closed (rename_expr r e)) true) as [ | Hopen];
    [ reflexivity | ].
  exfalso. apply Hopen. rewrite code_closed_renamed. exact Hclosed.
Qed.

Theorem closed_expr_equal : closed_expr_equal_statement.
Proof.
  intros [e Hclosed] [e' Hclosed']. simpl. intros <-. f_equal.
  apply UIP_dec, bool_dec.
Qed.

(* The free names of a closed expression, as the relations have them. *)
Lemma closed_var_free (e : closed_expr) (x : variable) :
  var_free_in_expr x (proj1_sig e)
  <-> among x (free_variables_of_expr (proj1_sig e)) = true.
Proof. apply var_free_agrees_expr, (proj2_sig e). Qed.

Lemma closed_cont_free (e : closed_expr) (k : continuation) :
  cont_free_in_expr k (proj1_sig e)
  <-> among k (free_continuations_of_expr (proj1_sig e)) = true.
Proof. apply cont_free_agrees_expr, (proj2_sig e). Qed.

Lemma alpha_equivalent_closed_refl (e : closed_expr) :
  alpha_equivalent_closed e e.
Proof. apply Alpha_refl. Qed.

Lemma alpha_equivalent_closed_sym (e e' : closed_expr) :
  alpha_equivalent_closed e e' -> alpha_equivalent_closed e' e.
Proof. apply Alpha_sym. Qed.

Lemma alpha_equivalent_closed_trans (e e' e'' : closed_expr) :
  alpha_equivalent_closed e e' -> alpha_equivalent_closed e' e'' ->
  alpha_equivalent_closed e e''.
Proof. apply Alpha_trans. Qed.

Lemma alpha_equivalent_closed_renamed (r : renaming) (e e' : closed_expr) :
  alpha_equivalent_closed e e' ->
  alpha_equivalent_closed (rename_closed_expr r e) (rename_closed_expr r e').
Proof.
  unfold alpha_equivalent_closed. rewrite !rename_closed_expr_term.
  apply alpha_equivalent_equivariant.
Qed.

Lemma bool_eq_of_iff (b b' : bool) : (b = true <-> b' = true) -> b = b'.
Proof. intros Hiff. apply eq_iff_eq_true, Hiff. Qed.

Theorem closed_expr_variables_nominal :
  closed_expr_variables_nominal_statement.
Proof.
  constructor; simpl.
  - exact alpha_equivalent_closed_refl.
  - exact alpha_equivalent_closed_sym.
  - exact alpha_equivalent_closed_trans.
  - intros e. apply closed_expr_equal. rewrite rename_closed_expr_term.
    apply rename_expr_id.
  - intros second first e. apply closed_expr_equal.
    rewrite !rename_closed_expr_term.
    exact (rename_expr_compose (variables_renaming second)
             (variables_renaming first) (proj1_sig e)).
  - intros p p' e Hsame. apply closed_expr_equal.
    rewrite !rename_closed_expr_term. apply rename_expr_ext.
    + intros x. apply Hsame.
    + reflexivity.
  - intros p e e'. apply alpha_equivalent_closed_renamed.
  - intros p e name. rewrite rename_closed_expr_term.
    rewrite free_variables_of_expr_renamed.
    exact (among_map_injective (rename_variable (variables_renaming p))
             (rename_variable_eqb (variables_renaming p)) name
             (free_variables_of_expr (proj1_sig e))).
  - intros e e' Halpha name. apply bool_eq_of_iff.
    rewrite <- !closed_var_free.
    exact (proj1 (alpha_equivalent_free_names _ _ Halpha) name).
  - intros p e Hfixes. unfold alpha_equivalent_closed.
    rewrite rename_closed_expr_term. apply Alpha_sym, Alpha_rename_by_relation.
    + exact (proj2_sig e).
    + intros x Hx. apply Hfixes, closed_var_free, Hx.
    + reflexivity.
Qed.

Theorem closed_expr_continuations_nominal :
  closed_expr_continuations_nominal_statement.
Proof.
  constructor; simpl.
  - exact alpha_equivalent_closed_refl.
  - exact alpha_equivalent_closed_sym.
  - exact alpha_equivalent_closed_trans.
  - intros e. apply closed_expr_equal. rewrite rename_closed_expr_term.
    apply rename_expr_id.
  - intros second first e. apply closed_expr_equal.
    rewrite !rename_closed_expr_term.
    exact (rename_expr_compose (continuations_renaming second)
             (continuations_renaming first) (proj1_sig e)).
  - intros p p' e Hsame. apply closed_expr_equal.
    rewrite !rename_closed_expr_term. apply rename_expr_ext.
    + reflexivity.
    + intros k. apply Hsame.
  - intros p e e'. apply alpha_equivalent_closed_renamed.
  - intros p e name. rewrite rename_closed_expr_term.
    rewrite free_continuations_of_expr_renamed.
    exact (among_map_injective
             (rename_continuation (continuations_renaming p))
             (rename_continuation_eqb (continuations_renaming p)) name
             (free_continuations_of_expr (proj1_sig e))).
  - intros e e' Halpha name. apply bool_eq_of_iff.
    rewrite <- !closed_cont_free.
    exact (proj2 (alpha_equivalent_free_names _ _ Halpha) name).
  - intros p e Hfixes. unfold alpha_equivalent_closed.
    rewrite rename_closed_expr_term. apply Alpha_sym, Alpha_rename_by_relation.
    + exact (proj2_sig e).
    + reflexivity.
    + intros k Hk. apply Hfixes, closed_cont_free, Hk.
Qed.

(* ======================================================================== *)
(* The binders                                                              *)
(* ======================================================================== *)

(* A name that is free in the scope and is not bound is free in the
   abstraction. *)
Lemma among_without_cons (name bound_name : positive)
    (names : list positive) :
  name <> bound_name -> among name names = true ->
  among name (without [bound_name] names) = true.
Proof.
  intros Hne Hname. rewrite among_without, Hname.
  apply not_among_singleton in Hne. rewrite Hne. reflexivity.
Qed.

Lemma among_without_list (name : positive)
    (bound_names names : list positive) :
  name ∉ bound_names -> among name names = true ->
  among name (without bound_names names) = true.
Proof.
  intros Hnotin Hname. rewrite among_without, Hname.
  apply not_among_in in Hnotin. rewrite Hnotin. reflexivity.
Qed.

(* One side of each binder, opened by a permutation that fixes the free
   names of the abstraction. *)
Lemma let_opened (p : permutation) (kind : flambda_kind) (d : named)
    (x : variable) (body : closed_expr) :
  fixes p (without [x] (free_variables_of_expr (proj1_sig body))) ->
  alpha_equivalent (Let x kind d (proj1_sig body))
    (Let (permute p x) kind d
       (rename_expr (variables_renaming p) (proj1_sig body))).
Proof.
  intros Hfixes.
  apply (Alpha_let_by_relation (variables_renaming p) x kind d
           (proj1_sig body)).
  - exact (proj2_sig body).
  - intros y Hne Hy. apply Hfixes, among_without_cons; [ exact Hne | ].
    apply closed_var_free, Hy.
  - reflexivity.
Qed.

Lemma let_cont_opened (p : permutation) (h : continuation_handler)
    (k : continuation) (body : closed_expr) :
  fixes p (without [k] (free_continuations_of_expr (proj1_sig body))) ->
  alpha_equivalent (Let_cont_non_recursive k h (proj1_sig body))
    (Let_cont_non_recursive (permute p k) h
       (rename_expr (continuations_renaming p) (proj1_sig body))).
Proof.
  intros Hfixes.
  apply (Alpha_let_cont_by_relation (continuations_renaming p) k h
           (proj1_sig body)).
  - exact (proj2_sig body).
  - reflexivity.
  - intros k' Hne Hk'. apply Hfixes, among_without_cons; [ exact Hne | ].
    apply closed_cont_free, Hk'.
Qed.

Lemma handler_opened (p : permutation) (k : continuation)
    (metadata : handler_metadata) (body : expr) (params : list variable)
    (handler : closed_expr) :
  fixes p (without params (free_variables_of_expr (proj1_sig handler))) ->
  alpha_equivalent
    (Let_cont_non_recursive k (Handler params metadata (proj1_sig handler))
       body)
    (Let_cont_non_recursive k
       (Handler (map (permute p) params) metadata
          (rename_expr (variables_renaming p) (proj1_sig handler)))
       body).
Proof.
  intros Hfixes.
  apply (Alpha_handler_by_relation (variables_renaming p) k params metadata
           (proj1_sig handler) body).
  - exact (proj2_sig handler).
  - intros y Hnotin Hy. apply Hfixes, among_without_list; [ exact Hnotin | ].
    apply closed_var_free, Hy.
  - reflexivity.
Qed.

Theorem let_abstraction : let_abstraction_statement.
Proof.
  intros kind d [x body] [x' body']
    (p & p' & Hopens & Hopens' & Hbound & Hscope).
  unfold opens, free_names_abstraction in Hopens, Hopens'.
  simpl in Hopens, Hopens', Hbound, Hscope. simpl.
  unfold alpha_equivalent_closed in Hscope.
  rewrite !rename_closed_expr_term in Hscope.
  eapply Alpha_trans; [ exact (let_opened p kind d x body Hopens) | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym; exact (let_opened p' kind d x' body' Hopens') ].
  rewrite Hbound.
  exact (Alpha_context (Let_body (permute p' x') kind d Hole) _ _ Hscope).
Qed.

Theorem let_cont_abstraction : let_cont_abstraction_statement.
Proof.
  intros h [k body] [k' body'] (p & p' & Hopens & Hopens' & Hbound & Hscope).
  unfold opens, free_names_abstraction in Hopens, Hopens'.
  simpl in Hopens, Hopens', Hbound, Hscope. simpl.
  unfold alpha_equivalent_closed in Hscope.
  rewrite !rename_closed_expr_term in Hscope.
  eapply Alpha_trans; [ exact (let_cont_opened p h k body Hopens) | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym; exact (let_cont_opened p' h k' body' Hopens') ].
  rewrite Hbound.
  exact (Alpha_context
           (Let_cont_non_recursive_body (permute p' k') h Hole) _ _ Hscope).
Qed.

Theorem handler_abstraction : handler_abstraction_statement.
Proof.
  intros k metadata body [params handler] [params' handler']
    (p & p' & Hopens & Hopens' & Hbound & Hscope).
  unfold opens, free_names_abstraction in Hopens, Hopens'.
  simpl in Hopens, Hopens', Hbound, Hscope. simpl.
  unfold alpha_equivalent_closed in Hscope.
  rewrite !rename_closed_expr_term in Hscope.
  eapply Alpha_trans;
    [ exact (handler_opened p k metadata body params handler Hopens) | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym;
        exact (handler_opened p' k metadata body params' handler' Hopens') ].
  rewrite Hbound.
  exact (Alpha_context
           (Let_cont_non_recursive_handler k (map (permute p') params')
              metadata Hole body)
           _ _ Hscope).
Qed.

(* Code. Closed code may be renamed by any renaming, so opening needs
   nothing of the permutation. *)
Lemma rename_variables_continuations_renaming (p : permutation)
    (xs : list variable) :
  map (rename_variable (continuations_renaming p)) xs = xs.
Proof.
  induction xs as [ | x xs IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma code_variables_opened (p : permutation) (before : list static_binding)
    (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (code_body : expr)
    (after : list static_binding) (body : expr) :
  code_is_closed (Code params my_closure ret exn metadata code_body) = true ->
  alpha_equivalent
    (Let_static
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body)
        :: after)
       body)
    (Let_static
       (before
        ++ Bind_code id
             (Code (map (permute p) params) (permute p my_closure) ret exn
                metadata (rename_expr (variables_renaming p) code_body))
        :: after)
       body).
Proof.
  intros Hclosed.
  exact (Alpha_code (variables_renaming p) before id
           (Code params my_closure ret exn metadata code_body) after body
           Hclosed).
Qed.

Lemma code_continuations_opened (p : permutation)
    (before : list static_binding) (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (code_body : expr)
    (after : list static_binding) (body : expr) :
  code_is_closed (Code params my_closure ret exn metadata code_body) = true ->
  alpha_equivalent
    (Let_static
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body)
        :: after)
       body)
    (Let_static
       (before
        ++ Bind_code id
             (Code params my_closure (permute p ret) (permute p exn)
                metadata (rename_expr (continuations_renaming p) code_body))
        :: after)
       body).
Proof.
  intros Hclosed.
  pose proof
    (Alpha_code (continuations_renaming p) before id
       (Code params my_closure ret exn metadata code_body) after body
       Hclosed)
    as Halpha.
  simpl in Halpha.
  rewrite rename_variables_continuations_renaming in Halpha.
  exact Halpha.
Qed.

Theorem code_variables_abstraction : code_variables_abstraction_statement.
Proof.
  intros before id ret exn metadata [bound_names code_body]
    [bound_names' code_body'] c c' after body Hc Hc' Hclosed Hclosed'
    (p & p' & _ & _ & Hbound & Hscope).
  unfold code_of_variables in Hc, Hc'. simpl in Hc, Hc', Hbound, Hscope.
  destruct bound_names as [ | my_closure params]; [ discriminate | ].
  destruct bound_names' as [ | my_closure' params']; [ discriminate | ].
  injection Hc as <-. injection Hc' as <-.
  simpl in Hbound. injection Hbound as Hmy_closure Hparams.
  unfold alpha_equivalent_closed in Hscope.
  rewrite !rename_closed_expr_term in Hscope.
  eapply Alpha_trans;
    [ exact (code_variables_opened p before id params my_closure ret exn
               metadata (proj1_sig code_body) after body Hclosed)
    | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym;
        exact (code_variables_opened p' before id params' my_closure' ret
                 exn metadata (proj1_sig code_body') after body Hclosed') ].
  rewrite Hmy_closure, Hparams.
  exact (Alpha_context
           (Let_static_code before id (map (permute p') params')
              (permute p' my_closure') ret exn metadata Hole after body)
           _ _ Hscope).
Qed.

Theorem code_continuations_abstraction :
  code_continuations_abstraction_statement.
Proof.
  intros before id params my_closure metadata [bound_names code_body]
    [bound_names' code_body'] c c' after body Hc Hc' Hclosed Hclosed'
    (p & p' & _ & _ & Hbound & Hscope).
  unfold code_of_continuations in Hc, Hc'.
  simpl in Hc, Hc', Hbound, Hscope.
  destruct bound_names as [ | ret [ | exn [ | ]]]; try discriminate.
  destruct bound_names' as [ | ret' [ | exn' [ | ]]]; try discriminate.
  injection Hc as <-. injection Hc' as <-.
  simpl in Hbound. injection Hbound as Hret Hexn.
  unfold alpha_equivalent_closed in Hscope.
  rewrite !rename_closed_expr_term in Hscope.
  eapply Alpha_trans;
    [ exact (code_continuations_opened p before id params my_closure ret exn
               metadata (proj1_sig code_body) after body Hclosed)
    | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym;
        exact (code_continuations_opened p' before id params my_closure ret'
                 exn' metadata (proj1_sig code_body') after body Hclosed') ].
  rewrite Hret, Hexn.
  exact (Alpha_context
           (Let_static_code before id params my_closure (permute p' ret')
              (permute p' exn') metadata Hole after body)
           _ _ Hscope).
Qed.
