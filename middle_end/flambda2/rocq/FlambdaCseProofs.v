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

(* Proofs of the statements of FlambdaCseSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaBindersSpec
  FlambdaNominalSpec FlambdaNominalProofs
  FlambdaApplyProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaCseSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what mentions it is forgotten. *)
Lemma all_computed_forget (known : computed) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  all_computed known vars statics ->
  all_computed (forget_computed x known) ((x, v) :: vars) statics.
Proof.
  intros Hhold entry Hin. unfold forget_computed in Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey Hargs].
  apply negb_true_iff, Pos.eqb_neq in Hkey. apply negb_true_iff in Hargs.
  destruct (Hhold entry Hin) as (vs & r & Hvs & Hprim & Hlookup).
  exists vs, r. split; [ | split; [ exact Hprim | ] ].
  - rewrite eval_simples_extend; [ exact Hvs | ].
    intros Hfree. apply var_in_simples_sound in Hfree.
    exact (eq_true_false_abs _ Hfree Hargs).
  - simpl. destruct (Pos.eqb_spec (fst entry) x) as [Heq | _];
      [ | exact Hlookup ].
    exfalso. apply Hkey. symmetry. exact Heq.
Qed.

Lemma all_computed_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : computed) (vs : list value) (vars handler_vars : env),
    all_computed known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    all_computed (forget_computed_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, all_computed_forget, Hhold.
Qed.

(* A static environment that binds more: a primitive has the value it
   had. *)
Lemma all_computed_extend (known : computed) (vars : env)
    (statics statics' : static_env) :
  all_computed known vars statics -> statics_extend statics statics' ->
  all_computed known vars statics'.
Proof.
  intros Hhold [_ Hvalues] entry Hin.
  destruct (Hhold entry Hin) as (vs & v & Hvs & Hprim & Hlookup).
  exists vs, v. split; [ exact Hvs | split; [ | exact Hlookup ] ].
  apply (eval_prim_statics_extend statics statics' _ vs v);
    [ | exact Hprim ].
  intros s allocated _ Hs. apply Hvalues, Hs.
Qed.

(* Where an untagging has a value, the tagging of that value is what
   was untagged. *)
Lemma tag_of_untagged (statics : static_env) (tagged v : value) :
  eval_prim statics (Unary Untag_immediate) [tagged] = Some v ->
  eval_prim statics (Unary Tag_immediate) [v] = Some tagged.
Proof.
  rewrite !eval_prim_unary. unfold eval_unary_primitive, int_of_value.
  destruct tagged; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  simpl.
  match goal with
  | |- context [in_range ?w ?i] =>
    destruct (in_range w i) eqn:Hrange; [ | discriminate ]
  end.
  intros [= <-]. simpl. rewrite Hrange. reflexivity.
Qed.

(* What [tagged_again] adds is of the one form. *)
Lemma tagged_again_spec (x : variable) (p : prim) (args : list simple)
    (entry : variable * (prim * list simple)) :
  In entry (tagged_again x p args) ->
  ∃ t : variable,
    p = Unary Untag_immediate
    ∧ args = [Var t]
    ∧ entry = (t, (Unary Tag_immediate, [Var x])).
Proof.
  unfold tagged_again.
  destruct p as [u | | | | ]; try contradiction.
  destruct u; try contradiction.
  destruct args as [ | [t | s | c] [ | arg args]]; try contradiction.
  intros [<- | []]. exists t. auto.
Qed.

(* The variable that it says something of is one of the arguments. *)
Lemma tagged_again_free (x : variable) (p : prim) (args : list simple)
    (entry : variable * (prim * list simple)) :
  In entry (tagged_again x p args) ->
  In (fst entry) (free_variables_of_named (Prim p args)).
Proof.
  intros Hin.
  destruct (tagged_again_spec x p args entry Hin) as (t & -> & -> & ->).
  left. reflexivity.
Qed.

(* Under a [Let]. *)
Lemma all_computed_under_let (known : computed) (x : variable) (d : named)
    (vars : env) (statics : static_env) (v : value) :
  all_computed known vars statics ->
  eval_named statics vars d = Some v ->
  all_computed (computed_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold computed_under_let.
  pose proof (all_computed_forget known vars statics x v Hhold) as Hforgot.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hforgot.
  destruct (shareable p && negb (var_in_simples x args)) eqn:Hadd;
    [ | exact Hforgot ].
  apply andb_true_iff in Hadd as [_ Hmention].
  apply negb_true_iff in Hmention.
  simpl in Hd.
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | discriminate ].
  simpl in Hd.
  intros entry [<- | Hin].
  - exists vs, v. simpl. split; [ | split; [ exact Hd | ] ].
    + rewrite eval_simples_extend; [ exact Hvs | ].
      intros Hfree. apply var_in_simples_sound in Hfree. congruence.
    + rewrite Pos.eqb_refl. reflexivity.
  - apply in_app_or in Hin as [Hagain | Hin]; [ | exact (Hforgot entry Hin) ].
    destruct (tagged_again_spec x p args entry Hagain) as (t & -> & -> & ->).
    assert (Hxt : Pos.eqb x t = false).
    { cbn [var_in_simples var_in_simple existsb] in Hmention.
      rewrite orb_false_r in Hmention. exact Hmention. }
    cbn [eval_simples eval_simple] in Hvs.
    destruct (lookup t vars) as [tagged | ] eqn:Ht; [ | discriminate ].
    simpl in Hvs. injection Hvs as <-.
    exists [v], tagged. cbn [fst snd].
    split; [ | split ].
    + cbn [eval_simples eval_simple lookup]. rewrite Pos.eqb_refl.
      reflexivity.
    + exact (tag_of_untagged statics tagged v Hd).
    + cbn [lookup]. rewrite Pos.eqb_sym, Hxt. exact Ht.
Qed.

(* ======================================================================== *)
(* Sharing                                                                  *)
(* ======================================================================== *)

Lemma computed_by_sound (known : computed) (p : prim) (args : list simple)
    (x : variable) :
  computed_by known p args = Some x -> (x, (p, args)) ∈ known.
Proof.
  induction known as [ | [y [p' args']] known IH]; simpl; [ discriminate | ].
  destruct (prim_eqb p p' && same_simples args args') eqn:Hsame.
  - intros [= ->]. apply andb_true_iff in Hsame as [Hp Hargs].
    apply prim_eqb_sound in Hp. unfold same_simples in Hargs.
    destruct (list_eq_dec simple_eq_dec args args') as [Heq | ];
      [ | discriminate ].
    subst. left. reflexivity.
  - intros Hcomputed. right. apply IH, Hcomputed.
Qed.

Lemma share_named_sound (known : computed) (vars : env)
    (statics : static_env) (d : named) (v : value) :
  all_computed known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (share_named known d) = Some v.
Proof.
  intros Hhold Hd. unfold share_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (computed_by known p args) as [x | ] eqn:Hcomputed;
    [ | exact Hd ].
  destruct (Hhold _ (computed_by_sound _ _ _ _ Hcomputed))
    as (vs & r & Hvs & Hprim & Hlookup).
  simpl in Hvs, Hprim, Hlookup. simpl in Hd. rewrite Hvs in Hd.
  simpl in Hd. rewrite Hprim in Hd. injection Hd as <-.
  simpl. exact Hlookup.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* As in FlambdaUnboxProofs.v: at a [Let], the variable has the value of
   the primitive or the primitive has none; then the parts under
   binders, by the induction, with what is known there. *)
Fixpoint share_primitives_knowing_refines (e : expr) :
  ∀ (known : computed) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    all_computed known vars statics ->
    refines (State e vars conts statics hp)
      (State (share_primitives_knowing known e) vars conts statics hp)
with share_primitives_in_handler_refines (h : continuation_handler) :
  ∀ (known : computed) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    all_computed known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp)
      (State
         (Let_cont_non_recursive k (share_primitives_in_handler known h)
            body)
         vars conts statics hp).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp Hhold; simpl.
    + (* Let *)
      destruct (eval_named statics vars d) as [v | ] eqn:Hd;
        [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
      * eapply refines_step;
          [ rewrite step_let, Hd, Hkind; reflexivity
          | rewrite step_let,
              (share_named_sound known vars statics d v Hhold Hd), Hkind;
            reflexivity
          | ].
        apply share_primitives_knowing_refines.
        exact (all_computed_under_let known x d vars statics v Hhold Hd).
      * apply step_undefined_refines. rewrite step_let, Hd, Hkind.
        reflexivity.
      * apply step_undefined_refines. rewrite step_let, Hd. reflexivity.
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply share_primitives_knowing_refines.
      exact (all_computed_extend known vars statics extended Hhold
        (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (share_primitives_in_handler known h)
              body)
           vars conts statics hp)).
      * apply share_primitives_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          share_primitives_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply share_primitives_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp Hhold. simpl.
    apply (let_cont_handler_refines_where (statics_extend statics)
      (statics_extend_trans statics));
      [ apply statics_extend_refl | ].
    intros vs handler_vars conts' statics' hp' Hextend _ Hbind.
    apply share_primitives_knowing_refines.
    apply (all_computed_bind_params statics' params known vs vars
      handler_vars);
      [ exact (all_computed_extend known vars statics statics' Hhold
          Hextend)
      | exact Hbind ].
Qed.

Theorem share_known_primitives_refines :
  share_known_primitives_refines_statement.
Proof.
  intros e vars conts statics hp. apply share_primitives_knowing_refines.
  intros entry [].
Qed.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma var_in_simples_renamed (r : renaming) (x : variable)
    (args : list simple) :
  var_in_simples (rename_variable r x) (map (rename_simple r) args)
  = var_in_simples x args.
Proof.
  unfold var_in_simples.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite IH.
  destruct s as [y | sym | c]; simpl; [ rewrite rename_variable_eqb | | ];
    reflexivity.
Qed.

Lemma rename_simple_injective (r : renaming) (s s' : simple) :
  rename_simple r s = rename_simple r s' -> s = s'.
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl; intros H;
    inversion H; try reflexivity.
  f_equal. exact (permute_injective (variable_permutation r) x x' H1).
Qed.

Lemma rename_simples_injective (r : renaming) (args : list simple) :
  ∀ args' : list simple,
    map (rename_simple r) args = map (rename_simple r) args' -> args = args'.
Proof.
  induction args as [ | s args IH]; intros [ | s' args']; simpl; intros H;
    try discriminate H; [ reflexivity | ].
  inversion H as [[Hs Hargs]].
  rewrite (rename_simple_injective r s s' Hs), (IH args' Hargs).
  reflexivity.
Qed.

Lemma same_simples_renamed (r : renaming) (args args' : list simple) :
  same_simples (map (rename_simple r) args) (map (rename_simple r) args')
  = same_simples args args'.
Proof.
  unfold same_simples.
  destruct (list_eq_dec simple_eq_dec args args') as [-> | Hdiffer].
  - destruct
      (list_eq_dec simple_eq_dec (map (rename_simple r) args')
         (map (rename_simple r) args'))
      as [_ | Hne];
      [ reflexivity | contradiction Hne; reflexivity ].
  - destruct
      (list_eq_dec simple_eq_dec (map (rename_simple r) args)
         (map (rename_simple r) args'))
      as [Hsame | _];
      [ | reflexivity ].
    contradiction Hdiffer.
    exact (rename_simples_injective r args args' Hsame).
Qed.

Lemma computed_by_renamed (r : renaming) (known : computed) (p : prim)
    (args : list simple) :
  computed_by (rename_computed r known) p (map (rename_simple r) args)
  = option_map (rename_variable r) (computed_by known p args).
Proof.
  induction known as [ | [x [p' args']] known IH]; simpl; [ reflexivity | ].
  rewrite same_simples_renamed.
  destruct (prim_eqb p p' && same_simples args args');
    [ reflexivity | exact IH ].
Qed.

Lemma share_named_renamed (r : renaming) (known : computed) (d : named) :
  share_named (rename_computed r known) (rename_named r d)
  = rename_named r (share_named known d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  simpl. rewrite computed_by_renamed.
  destruct (computed_by known p args); reflexivity.
Qed.

Lemma forget_computed_renamed (r : renaming) (x : variable)
    (known : computed) :
  forget_computed (rename_variable r x) (rename_computed r known)
  = rename_computed r (forget_computed x known).
Proof.
  induction known as [ | [y [p args]] known IH]; [ reflexivity | ].
  change
    ((if
        negb (Pos.eqb (rename_variable r x) (rename_variable r y))
        && negb
             (var_in_simples (rename_variable r x)
                (map (rename_simple r) args))
      then
        (rename_variable r y, (p, map (rename_simple r) args))
        :: forget_computed (rename_variable r x) (rename_computed r known)
      else forget_computed (rename_variable r x) (rename_computed r known))
     = rename_computed r
         (if negb (Pos.eqb x y) && negb (var_in_simples x args)
          then (y, (p, args)) :: forget_computed x known
          else forget_computed x known)).
  rewrite rename_variable_eqb, var_in_simples_renamed, IH.
  destruct (negb (Pos.eqb x y) && negb (var_in_simples x args)); reflexivity.
Qed.

Lemma forget_computed_vars_renamed (r : renaming) (xs : list variable) :
  ∀ known : computed,
    forget_computed_vars (map (rename_variable r) xs)
      (rename_computed r known)
    = rename_computed r (forget_computed_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known; simpl; [ reflexivity | ].
  rewrite forget_computed_renamed. apply IH.
Qed.

Lemma tagged_again_renamed (r : renaming) (x : variable) (p : prim)
    (args : list simple) :
  tagged_again (rename_variable r x) p (map (rename_simple r) args)
  = map
      (fun entry =>
         (rename_variable r (fst entry),
          (fst (snd entry), map (rename_simple r) (snd (snd entry)))))
      (tagged_again x p args).
Proof.
  unfold tagged_again.
  destruct p as [u | | | | ]; try reflexivity.
  destruct u; try reflexivity.
  destruct args as [ | [t | s | c] [ | arg args]]; reflexivity.
Qed.

Lemma computed_under_let_renamed (r : renaming) (known : computed)
    (x : variable) (d : named) :
  computed_under_let (rename_computed r known) (rename_variable r x)
    (rename_named r d)
  = rename_computed r (computed_under_let known x d).
Proof.
  unfold computed_under_let. rewrite forget_computed_renamed.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  simpl. rewrite var_in_simples_renamed.
  destruct (shareable p && negb (var_in_simples x args)); [ | reflexivity ].
  unfold rename_computed. rewrite map_cons, map_app. cbn [fst snd].
  f_equal. f_equal. apply tagged_again_renamed.
Qed.

Fixpoint share_primitives_knowing_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : computed,
    share_primitives_knowing (rename_computed r known) (rename_expr r e)
    = rename_expr r (share_primitives_knowing known e)
with share_primitives_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : computed,
    share_primitives_in_handler (rename_computed r known)
      (rename_handler r h)
    = rename_handler r (share_primitives_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; simpl; try reflexivity.
    + rewrite share_named_renamed, computed_under_let_renamed,
        share_primitives_knowing_equivariant_aux.
      reflexivity.
    + rewrite share_primitives_knowing_equivariant_aux. reflexivity.
    + rewrite share_primitives_in_handler_equivariant,
        share_primitives_knowing_equivariant_aux.
      reflexivity.
    + rewrite share_primitives_knowing_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known. simpl.
    rewrite forget_computed_vars_renamed,
      share_primitives_knowing_equivariant_aux.
    reflexivity.
Qed.

Theorem share_primitives_knowing_equivariant :
  share_primitives_knowing_equivariant_statement.
Proof.
  intros r known e. apply share_primitives_knowing_equivariant_aux.
Qed.

Theorem share_known_primitives_equivariant :
  share_known_primitives_equivariant_statement.
Proof.
  intros r e. exact (share_primitives_knowing_equivariant_aux r e []).
Qed.
