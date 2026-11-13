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


(* Proofs of the statements of FlambdaLiftObservedSpec.v. Nothing here
   needs review.

   The simulation of each lifting is the one of its own proofs file,
   from the state that the unit starts in with a continuation
   environment of what may follow a unit
   (FlambdaObserversSpec.observers), and without the invariant of the
   first run that the test of the exits gives. What may follow a unit
   has no [Let] to lift and no code, so each rewrite leaves it as it is
   and it is related to itself; it binds nothing static, so the run of
   the unit lifted from it makes no static fault
   (FlambdaUnitFromProofs.well_formed_never_faults_under); and it names
   no code id, so it makes no closure of the code of a function. The
   values of two exits that answer each other are lifted, so they have
   the same constants and the same locations
   (FlambdaLiftProofs.lifted_same_scalars), and the two heaps are the
   same.

   The lifting to a group around needs, of the run of the unit
   rewritten, that it reads no variable that has no value, which is so
   of a unit run on its own and not of what may follow it. Its
   statement is proved from the one about the two exits
   (FlambdaObserversSpec.under_observers_worlds_statement): the two
   units are run on their own to their exits, as in
   FlambdaLiftAroundProofs.v, and what follows is the same term in the
   two runs, with no group put (FlambdaLiftAroundProofs.unit_around_worlds).

   The tables bound at the head after the four liftings: the statement
   of that step is of a term run under continuations in which the
   symbols of the tables do not occur
   (FlambdaUnusedHeadStaticsSpec.add_head_tables_refines_under_statement).
   What may follow a unit names no symbol and no code id and binds
   nothing static, so no set of names occurs in it
   ([observers_free_of]).

   The copies of code after the tables: their statement is of a unit
   that something follows
   (FlambdaCopyCodeSpec.with_copies_refines_observed_statement), and
   asks nothing of the imports; the unit under the copies binds no
   more than the unit with them
   (FlambdaCopyCodeProofs.free_in_under_copies). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaImportsSpec
  FlambdaImportsProofs FlambdaStaticsInScopeSpec FlambdaUnusedStaticsSpec
  FlambdaUnusedHeadStaticsSpec FlambdaUnusedHeadStaticsProofs
  FlambdaUnitFromProofs
  FlambdaObserversSpec FlambdaObserversProofs FlambdaMyClosureSpec
  FlambdaLiftSpec FlambdaLiftProofs FlambdaLiftFromCodeSpec
  FlambdaLiftFromCodeProofs FlambdaLiftFromArmsSpec FlambdaLiftFromArmsProofs
  FlambdaLiftAroundSpec FlambdaLiftAroundProofs FlambdaLiftObservedSpec.
From OxCamlFlambda2 Require Import FlambdaCopyCodeSpec.
From OxCamlFlambda2 Require FlambdaCopyCodeProofs.
From OxCamlFlambda2 Require FlambdaValidateImportsProofs.
Import ListNotations.

Theorem lift_to_head_unit_observed_refines :
  lift_to_head_unit_observed_refines_statement.
Proof.
  intros imports u lets e e'. unfold lift_to_head_unit_observed. cbv zeta.
  destruct (well_formed_unit u (lift_to_head lets e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-].
  destruct (once_each [] (lifted_out_bindings lets root e))
    as [ | binding rest] eqn:Hbindings.
  - unfold lift_to_head. cbv zeta. rewrite Hbindings. intros _ conts _.
    apply observed_refines_refl.
  - assert (Hsome : once_each [] (lifted_out_bindings lets root e) <> [])
      by (rewrite Hbindings; discriminate).
    assert (Hshape :
      lift_to_head lets e
      = Let_static (once_each [] (lifted_out_bindings lets root e))
          (lift_from_code [] lets root e)).
    { unfold lift_to_head. cbv zeta. rewrite Hbindings. reflexivity. }
    rewrite Hshape in Hwf |- *. intros Hfree conts Hobservers.
    exact (FlambdaLiftFromArmsProofs.unit_lifted_observed u lets e
             (imports_statics imports) empty_heap conts Hwf Hsome eq_refl
             Hfree Hobservers).
Qed.

Theorem lift_from_code_unit_observed_refines :
  lift_from_code_unit_observed_refines_statement.
Proof.
  intros imports u K e e'. unfold lift_from_code_unit_observed. cbv zeta.
  destruct (well_formed_unit u (lift_from_code K [] root e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-] Hfree conts Hobservers.
  exact (unit_lifted_out_observed K u e (imports_statics imports) empty_heap
           conts Hwf eq_refl Hfree Hobservers).
Qed.

Theorem lift_unit_observed_refines : lift_unit_observed_refines_statement.
Proof.
  intros imports L e e'. unfold lift_unit_observed. cbv zeta.
  fold (lifted_unit (lifted_my_closures L) (lifted_lets L) e).
  destruct
    (well_formed_unit (lifting_unit L)
       (lifted_unit (lifted_my_closures L) (lifted_lets L) e))
    eqn:Hwf;
    simpl; [ | discriminate ].
  destruct
    (only_closures_of (lifted_my_closures L)
       (lifted_unit (lifted_my_closures L) (lifted_lets L) e))
    eqn:Honly;
    [ | discriminate ].
  intros [= <-] Hfree conts Hobservers.
  exact (FlambdaLiftProofs.unit_lifted_observed (lifted_my_closures L)
           (lifting_unit L) (lifted_lets L) e (imports_statics imports)
           empty_heap conts Hwf Honly eq_refl Hfree
           (fun id sym _ => statics_ok_imports id sym imports) Hobservers).
Qed.

Theorem lift_around_unit_observed_refines :
  lift_around_unit_observed_refines_statement.
Proof.
  intros imports u lets places e e'. unfold lift_around_unit_observed.
  cbv zeta.
  set (groups := groups_of lets places e).
  destruct (around_ok lets groups [] root e) eqn:Hok; simpl;
    [ | discriminate ].
  destruct (well_formed_unit u (lift_around lets groups root e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-] Hfree conts Hobservers.
  exact (unit_around_observed u lets groups e (imports_statics imports)
           empty_heap conts Hwf Hok eq_refl Hfree Hobservers).
Qed.

(* A lifting that is made where its conditions hold, and the term as it
   is where they do not. *)
Lemma made_or_same (statics : static_env) (hp : heap)
    (lifted : option expr) (e : expr) :
  (∀ e' : expr,
    lifted = Some e' -> refines_under_observers [] statics hp e e') ->
  refines_under_observers [] statics hp e
    match lifted with
    | Some e' => e'
    | None => e
    end.
Proof.
  intros H. destruct lifted as [e' | ]; [ exact (H e' eq_refl) | ].
  intros conts _. apply observed_refines_refl.
Qed.

Theorem liftings_observed_refines : liftings_observed_refines_statement.
Proof.
  intros imports L e. cbv zeta. intros Hhead Hcode Hlets Haround.
  unfold liftings_observed in *.
  refine
    (refines_under_observers_trans [] _ _ e (to_head_observed L e) _ _ _).
  { unfold to_head_observed in *. apply made_or_same. intros e' Hstep.
    rewrite Hstep in Hhead.
    exact (lift_to_head_unit_observed_refines imports _ _ e e' Hstep Hhead). }
  refine
    (refines_under_observers_trans [] _ _ _
       (out_of_code_observed L (to_head_observed L e)) _ _ _).
  { unfold out_of_code_observed in *. apply made_or_same. intros e' Hstep.
    rewrite Hstep in Hcode.
    exact (lift_from_code_unit_observed_refines imports _ _ _ e' Hstep
             Hcode). }
  refine
    (refines_under_observers_trans [] _ _ _
       (lets_observed L (out_of_code_observed L (to_head_observed L e))) _ _
       _).
  { unfold lets_observed in *. apply made_or_same. intros e' Hstep.
    rewrite Hstep in Hlets.
    exact (lift_unit_observed_refines imports L _ e' Hstep Hlets). }
  unfold around_observed in *. apply made_or_same. intros e' Hstep.
  rewrite Hstep in Haround.
  exact (lift_around_unit_observed_refines imports _ _ _ _ e' Hstep
           Haround).
Qed.

(* ======================================================================== *)
(* What may follow a unit names nothing static                              *)
(* ======================================================================== *)

Section Free_of.

Variable D : dropped.

Lemma simple_nowhere_avoids (s : simple) :
  simple_statics_in_scope [] s = true -> simple_avoids D s = true.
Proof.
  destruct s as [x | sym | c]; simpl;
    [ reflexivity | discriminate | reflexivity ].
Qed.

Lemma simples_nowhere_avoid (args : list simple) :
  simples_statics_in_scope [] args = true -> simples_avoid D args = true.
Proof.
  unfold simples_statics_in_scope, simples_avoid.
  induction args as [ | a args IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Ha Hargs].
  rewrite (simple_nowhere_avoids a Ha). exact (IH Hargs).
Qed.

Lemma value_slots_nowhere_avoid (slots : list (value_slot * simple)) :
  value_slots_statics_in_scope [] slots = true ->
  value_slots_avoid D slots = true.
Proof.
  unfold value_slots_statics_in_scope, value_slots_avoid.
  induction slots as [ | slot slots IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hslot Hslots].
  rewrite (simple_nowhere_avoids (snd slot) Hslot). exact (IH Hslots).
Qed.

Lemma named_nowhere_avoids (d : named) :
  named_statics_in_scope [] [] d = true -> named_avoids D d = true.
Proof.
  destruct d as [ s | p args | code slots | slot decls slots | | | ]; simpl.
  - apply simple_nowhere_avoids.
  - apply simples_nowhere_avoid.
  - discriminate.
  - intros H. apply andb_true_iff in H as [Hdecls Hslots].
    destruct decls as [ | decl decls];
      [ | simpl in Hdecls; discriminate Hdecls ].
    exact (value_slots_nowhere_avoid slots Hslots).
  - apply simple_nowhere_avoids.
  - apply simple_nowhere_avoids.
  - apply simple_nowhere_avoids.
Qed.

Lemma apply_nowhere_avoids (ap : apply_expr) :
  apply_statics_in_scope [] [] ap = true -> apply_avoids D ap = true.
Proof.
  unfold apply_statics_in_scope, apply_avoids. intros H.
  apply andb_true_iff in H as [H Hextra].
  apply andb_true_iff in H as [H Hargs].
  apply andb_true_iff in H as [Hcallee Hkind].
  rewrite (simple_nowhere_avoids _ Hcallee), (simples_nowhere_avoid _ Hargs),
    (simples_nowhere_avoid _ Hextra).
  destruct (ap_call_kind ap) as [code | ];
    [ simpl in Hkind; discriminate Hkind | reflexivity ].
Qed.

Lemma apply_cont_nowhere_avoids (ac : apply_cont) :
  apply_cont_statics_in_scope [] ac = true -> apply_cont_avoids D ac = true.
Proof.
  unfold apply_cont_statics_in_scope, apply_cont_avoids.
  apply simples_nowhere_avoid.
Qed.

(* A term with no static binding in it that names no symbol and no code
   id refers to none of the names. *)
Fixpoint observer_avoids (e : expr) {struct e} :
  binds_no_static e = true ->
  expr_statics_in_scope [] [] e = true ->
  expr_avoids D e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl.
  - intros Hno H. apply andb_true_iff in H as [Hd Hbody].
    rewrite (named_nowhere_avoids d Hd).
    exact (observer_avoids body Hno Hbody).
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros Hno H.
    apply andb_true_iff in Hno as [Hno_h Hno_b].
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hh.
    simpl in Hh.
    rewrite (observer_avoids handler Hno_h Hh),
      (observer_avoids body Hno_b Hb).
    reflexivity.
  - intros Hno H. apply andb_true_iff in Hno as [Hno_hs Hno_b].
    apply andb_true_iff in H as [Hhs Hb].
    rewrite (observer_avoids body Hno_b Hb), andb_true_r.
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hhs.
    simpl in Hhs.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hno_hs, Hhs.
    apply andb_true_iff in Hno_hs as [Hno_h Hno_rest].
    apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (observer_avoids handler Hno_h Hh).
    exact (IH Hno_rest Hrest).
  - intros _. apply apply_nowhere_avoids.
  - intros _. apply apply_cont_nowhere_avoids.
  - intros _ H. apply andb_true_iff in H as [Hscrutinee Harms].
    rewrite (simple_nowhere_avoids scrutinee Hscrutinee). simpl.
    induction arms as [ | arm arms IH]; simpl; [ reflexivity | ].
    simpl in Harms. apply andb_true_iff in Harms as [Harm Hrest].
    rewrite (apply_cont_nowhere_avoids (snd arm) Harm). exact (IH Hrest).
  - intros _ _. reflexivity.
Qed.

(* And it binds none of them. *)
Fixpoint no_static_binds_only_kept (e : expr) {struct e} :
  binds_no_static e = true -> binds_only_kept D e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - apply no_static_binds_only_kept.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_binds_only_kept handler Hh),
      (no_static_binds_only_kept body Hb).
    reflexivity.
  - intros H. apply andb_true_iff in H as [Hhs Hb].
    rewrite (no_static_binds_only_kept body Hb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (no_static_binds_only_kept handler Hh). exact (IH Hrest).
Qed.

Lemma observer_free_of (slots : list value_slot) (c : cont_value) :
  observer slots c -> cont_free_of D c = true.
Proof.
  destruct c
    as [ [params metadata body] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; reflexivity ].
  destruct vars; [ | contradiction ].
  destruct conts; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & Hscope & _).
  unfold handler_free_of, env_free_of. simpl.
  rewrite (observer_avoids body Hno Hscope),
    (no_static_binds_only_kept body Hno).
  reflexivity.
Qed.

Lemma observers_free_of (slots : list value_slot) (conts : cont_env) :
  observers slots conts ->
  forallb (fun '(_, c) => cont_free_of D c) conts = true.
Proof.
  unfold observers.
  induction conts as [ | [k c] conts IH]; simpl; [ intros _; reflexivity | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  rewrite (observer_free_of slots c Hc). exact (IH Hrest).
Qed.

End Free_of.

(* ======================================================================== *)
(* The tables bound at the head                                             *)
(* ======================================================================== *)

(* The tables bound at the head of a unit that something follows, from
   the imports. *)
Lemma add_head_tables_observed (imports : list (symbol * import_class))
    (tables : list (symbol * list simple)) (e : expr) :
  head_free_in (imports_statics imports) (add_head_tables tables e) = true ->
  refines_under_observers [] (imports_statics imports) empty_heap e
    (add_head_tables tables e).
Proof.
  intros Hhead conts Hobservers. apply refines_observed.
  apply add_head_tables_refines_under; [ | exact Hhead ].
  pose proof (imports_statics_free_of (tables_dropped tables) imports)
    as Hstatics.
  unfold envs_free_of in Hstatics |- *.
  apply andb_true_iff in Hstatics as [_ Hstatics].
  rewrite Hstatics,
    (observers_free_of (tables_dropped tables) [] conts Hobservers).
  reflexivity.
Qed.

(* The term under the tables binds no more than the term with them. *)
Lemma free_in_under_tables (statics : static_env)
    (tables : list (symbol * list simple)) (e : expr) :
  static_names_free_in statics (add_head_tables tables e) = true ->
  static_names_free_in statics e = true.
Proof.
  unfold add_head_tables.
  destruct tables as [ | table tables]; [ intros H; exact H | ].
  cbv zeta.
  match goal with
  | |- static_names_free_in _ (if ?c then _ else _) = true -> _ =>
    destruct c
  end;
    [ | intros H; exact H ].
  unfold static_names_free_in. cbn [static_names]. intros H.
  apply andb_true_iff in H as [Hsymbols Hcode].
  rewrite forallb_app in Hsymbols, Hcode.
  apply andb_true_iff in Hsymbols as [_ Hsymbols].
  apply andb_true_iff in Hcode as [_ Hcode].
  rewrite Hsymbols, Hcode. reflexivity.
Qed.

Theorem with_lifting_observed_refines :
  with_lifting_observed_refines_statement.
Proof.
  intros imports L e. cbv zeta. intros Hhead Hcode Hlets Hcopied.
  unfold with_lifting_observed in *.
  pose proof (FlambdaCopyCodeProofs.free_in_under_copies _ _ _ Hcopied)
    as Hall.
  pose proof (free_in_under_tables _ _ _ Hall) as Haround.
  refine
    (refines_under_observers_trans [] _ _ e
       (add_head_tables (tables_put L) (liftings_observed L e)) _ _
       (FlambdaCopyCodeProofs.with_copies_refines_under imports []
          (code_copied L) _)).
  refine
    (refines_under_observers_trans [] _ _ e (liftings_observed L e) _ _ _).
  - exact (liftings_observed_refines imports L e Hhead Hcode Hlets Haround).
  - apply add_head_tables_observed.
    exact (FlambdaValidateImportsProofs.static_names_free_in_head _ _ Hall).
Qed.
