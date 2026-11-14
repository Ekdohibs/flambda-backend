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


(* Proofs of the statements of FlambdaSimplifySwitchMergeSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia Morphisms Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaControlFlowSpec FlambdaControlFlowProofs FlambdaClosuresSpec
  FlambdaApplyProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs FlambdaCseSpec
  FlambdaCseProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetEnvProofs FlambdaTypeCreatorsSpec FlambdaTypeCreatorsProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetTermsSpec FlambdaMeetRenameSpec
  FlambdaMeetRenameProofs FlambdaSimplifySwitchSpec
  FlambdaSimplifySwitchProofs FlambdaSimplifySwitchArmsSpec
  FlambdaSimplifySwitchArmsProofs FlambdaSimplifySwitchMergeSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesProofs
  FlambdaMeetClosuresProofs FlambdaMeetTermsProofs FlambdaSimplifyPassSpec
  FlambdaSimplifyPassProofs.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The arithmetic of immediates                                             *)
(* ======================================================================== *)

(* Two integers that wrap to the same immediate. *)
Inductive congruent (a b : Z) : Prop :=
  | Congruent (H : wrap immediate_width a = wrap immediate_width b).

Lemma immediate_width_nonneg : 0 <= immediate_width.
Proof. vm_compute. discriminate. Qed.

Lemma immediate_width_positive : 1 <= immediate_width.
Proof. vm_compute. discriminate. Qed.

Lemma congruent_iff (a b : Z) :
  congruent a b <-> ∃ m : Z, a = b + m * 2 ^ immediate_width.
Proof.
  split.
  - intros [Hwrap].
    destruct (wrap_adds_multiple immediate_width a immediate_width_nonneg)
      as [k Hk].
    destruct (wrap_adds_multiple immediate_width b immediate_width_nonneg)
      as [k' Hk'].
    exists (k' - k). lia.
  - intros [m ->]. constructor. unfold wrap.
    replace (b + m * 2 ^ immediate_width + 2 ^ (immediate_width - 1))
      with (b + 2 ^ (immediate_width - 1) + m * 2 ^ immediate_width) by ring.
    rewrite Z_mod_plus_full. reflexivity.
Qed.

#[local] Instance congruent_equivalence : Equivalence congruent.
Proof.
  split.
  - intros a. constructor. reflexivity.
  - intros a b [H]. constructor. symmetry. exact H.
  - intros a b c [H] [H']. constructor. rewrite H. exact H'.
Qed.

#[local] Instance congruent_add :
  Proper (congruent ==> congruent ==> congruent) Z.add.
Proof.
  intros a a' Ha b b' Hb. apply congruent_iff in Ha as [m ->].
  apply congruent_iff in Hb as [m' ->]. apply congruent_iff.
  exists (m + m'). ring.
Qed.

#[local] Instance congruent_sub :
  Proper (congruent ==> congruent ==> congruent) Z.sub.
Proof.
  intros a a' Ha b b' Hb. apply congruent_iff in Ha as [m ->].
  apply congruent_iff in Hb as [m' ->]. apply congruent_iff.
  exists (m - m'). ring.
Qed.

#[local] Instance congruent_mul :
  Proper (congruent ==> congruent ==> congruent) Z.mul.
Proof.
  intros a a' Ha b b' Hb. apply congruent_iff in Ha as [m ->].
  apply congruent_iff in Hb as [m' ->]. apply congruent_iff.
  exists (m * b' + a' * m' + m * m' * 2 ^ immediate_width). ring.
Qed.

#[local] Instance congruent_opp : Proper (congruent ==> congruent) Z.opp.
Proof.
  intros a a' Ha. apply congruent_iff in Ha as [m ->]. apply congruent_iff.
  exists (- m). ring.
Qed.

#[local] Instance congruent_wrap_proper :
  Proper (congruent ==> congruent) (wrap immediate_width).
Proof. intros a a' [Ha]. rewrite Ha. reflexivity. Qed.

Lemma wrap_congruent (a : Z) : congruent (wrap immediate_width a) a.
Proof.
  constructor. apply wrap_of_in_range;
    [ exact immediate_width_positive | ].
  apply wrap_in_range, immediate_width_positive.
Qed.

(* An immediate that is congruent to an integer is what it wraps to. *)
Lemma congruent_wrap (i a : Z) :
  in_range immediate_width i = true -> congruent i a ->
  i = wrap immediate_width a.
Proof.
  intros Hrange [Hcongruent]. rewrite <- Hcongruent. symmetry.
  apply wrap_of_in_range; [ exact immediate_width_positive | exact Hrange ].
Qed.

Lemma immediate_wrapped (a : Z) :
  in_range immediate_width (wrap immediate_width a) = true.
Proof. apply wrap_in_range, immediate_width_positive. Qed.

(* The slope and the offset that are read off two arms give the
   arguments of both. *)
Lemma affine_of_two_arms (arm arg discr imm : Z) :
  in_range immediate_width arg = true ->
  in_range immediate_width imm = true ->
  Z.rem (immediate_sub imm arg) (immediate_sub discr arm) = 0 ->
  let slope :=
    immediate_div (immediate_sub imm arg) (immediate_sub discr arm)
  in
  let offset := immediate_sub arg (immediate_mul slope arm) in
  arg = immediate_add (immediate_mul arm slope) offset
  ∧ imm = immediate_add (immediate_mul discr slope) offset.
Proof.
  intros Harg Himm Hrem. cbv zeta.
  pose proof (Z.quot_rem' (immediate_sub imm arg) (immediate_sub discr arm))
    as Hquot.
  rewrite Hrem, Z.add_0_r in Hquot.
  set (q := Z.quot (immediate_sub imm arg) (immediate_sub discr arm)) in *.
  unfold immediate_add, immediate_mul, immediate_div.
  fold q.
  split; (apply congruent_wrap; [ assumption | ]).
  - unfold immediate_sub. rewrite !wrap_congruent.
    replace (arm * q + (arg - q * arm)) with arg by ring. reflexivity.
  - assert (Hdiff : congruent (imm - arg) ((discr - arm) * q)).
    { rewrite <- (wrap_congruent (imm - arg)).
      rewrite <- (wrap_congruent (discr - arm)).
      unfold immediate_sub in Hquot. rewrite Hquot at 1. reflexivity. }
    unfold immediate_sub. rewrite !wrap_congruent.
    replace (discr * q + (arg - q * arm)) with ((discr - arm) * q + arg)
      by ring.
    rewrite <- Hdiff. replace (imm - arg + arg) with imm by ring.
    reflexivity.
Qed.

(* ======================================================================== *)
(* What an argument that can be merged is, in an arm                        *)
(* ======================================================================== *)

Definition constant_of (kind : affine_immediate_kind) (i : Z) : simple :=
  Const (const_of_int (affine_immediate_kind_to_standard_int kind) i).

Definition affine_value (i offset slope : Z) : Z :=
  immediate_add (immediate_mul i slope) offset.

Definition affine_fits (kind : affine_immediate_kind) (offset slope : Z)
    (entry : Z * simple) : Prop :=
  snd entry = constant_of kind (affine_value (fst entry) offset slope).

Lemma recognize_affine_immediate_const_some (c : const)
    (kind : affine_immediate_kind) (imm : Z) :
  recognize_affine_immediate_const c = Some (kind, imm) ->
  Const c = constant_of kind imm ∧ in_range immediate_width imm = true.
Proof.
  destruct c; simpl; try discriminate;
    destruct (in_range immediate_width i) eqn:Hrange; try discriminate;
    intros [= <- <-]; (split; [ reflexivity | exact Hrange ]).
Qed.

Lemma affine_immediate_kind_eqb_eq (kind kind' : affine_immediate_kind) :
  affine_immediate_kind_eqb kind kind' = true -> kind = kind'.
Proof.
  destruct kind, kind'; simpl; intros H; try discriminate; reflexivity.
Qed.

(* What has been found of an argument, of the arms read so far. *)
Definition affine_seen (affine : affine_argument_of_mergeable_arm)
    (seen : list (Z * simple)) : Prop :=
  match affine with
  | Not_affine => True
  | Maybe_affine => seen = []
  | Affine_constant kind arm arg =>
    in_range immediate_width arg = true
    ∧ ∀ entry : Z * simple, entry ∈ seen -> entry = (arm, constant_of kind arg)
  | Affine kind offset slope =>
    in_range immediate_width offset = true
    ∧ in_range immediate_width slope = true
    ∧ ∀ entry : Z * simple, entry ∈ seen -> affine_fits kind offset slope entry
  end.

Lemma check_arm_seen (affine : affine_argument_of_mergeable_arm)
    (seen : list (Z * simple)) (entry : Z * simple) :
  affine_seen affine seen ->
  affine_seen (check_arm affine entry) (seen ++ [entry]).
Proof.
  intros Hseen. destruct entry as [discr s]. unfold check_arm. cbn [fst snd].
  destruct s as [x | sym | c]; try exact I.
  unfold recognize_affine_argument.
  destruct affine as [ | | kind arm arg | kind offset slope].
  - exact I.
  - destruct (recognize_affine_immediate_const c) as [[imm_kind imm] | ]
      eqn:Hc; [ | exact I ].
    apply recognize_affine_immediate_const_some in Hc as [Hc Hrange].
    simpl in Hseen. subst seen. split; [ exact Hrange | ].
    intros entry [<- | []]. rewrite Hc. reflexivity.
  - destruct (recognize_affine_immediate_const c) as [[imm_kind imm] | ]
      eqn:Hc; [ | exact I ].
    apply recognize_affine_immediate_const_some in Hc as [Hc Hrange].
    destruct Hseen as [Harg Hseen].
    match goal with
    | |- affine_seen (if ?test then _ else _) _ =>
      destruct test eqn:Htest; [ | exact I ]
    end.
    apply andb_true_iff in Htest as [Htest Hrem].
    apply andb_true_iff in Htest as [Hkind _].
    apply affine_immediate_kind_eqb_eq in Hkind. subst imm_kind.
    apply Z.eqb_eq in Hrem.
    destruct (affine_of_two_arms arm arg discr imm Harg Hrange Hrem)
      as [Hfirst Hsecond].
    split; [ apply immediate_wrapped | ].
    split; [ apply immediate_wrapped | ].
    intros entry Hin. apply in_app_iff in Hin as [Hin | [<- | []]].
    + rewrite (Hseen entry Hin). unfold affine_fits, affine_value.
      cbn [fst snd]. f_equal. exact Hfirst.
    + unfold affine_fits, affine_value. cbn [fst snd]. rewrite Hc. f_equal.
      exact Hsecond.
  - destruct (recognize_affine_immediate_const c) as [[imm_kind imm] | ]
      eqn:Hc; [ | exact I ].
    apply recognize_affine_immediate_const_some in Hc as [Hc Hrange].
    destruct Hseen as (Hoffset & Hslope & Hseen).
    match goal with
    | |- affine_seen (if ?test then _ else _) _ =>
      destruct test eqn:Htest; [ | exact I ]
    end.
    apply andb_true_iff in Htest as [Hkind Himm].
    apply affine_immediate_kind_eqb_eq in Hkind. subst imm_kind.
    apply Z.eqb_eq in Himm.
    split; [ exact Hoffset | ]. split; [ exact Hslope | ].
    intros entry Hin. apply in_app_iff in Hin as [Hin | [<- | []]].
    + exact (Hseen entry Hin).
    + unfold affine_fits, affine_value. cbn [fst snd]. rewrite Hc. f_equal.
      exact Himm.
Qed.

Lemma fold_check_arm_seen (arg : list (Z * simple)) :
  ∀ (affine : affine_argument_of_mergeable_arm) (seen : list (Z * simple)),
    affine_seen affine seen ->
    affine_seen (List.fold_left check_arm arg affine) (seen ++ arg).
Proof.
  induction arg as [ | entry arg IH]; intros affine seen Hseen; simpl.
  - rewrite app_nil_r. exact Hseen.
  - replace (seen ++ entry :: arg) with ((seen ++ [entry]) ++ arg)
      by (rewrite <- app_assoc; reflexivity).
    apply IH, check_arm_seen, Hseen.
Qed.

(* The value of an argument that can be merged, in the arm of a
   number. *)
Definition argument_value (vars : env) (i : Z) (argument : mergeable_argument)
    : option value :=
  match argument with
  | Invariant_argument s => eval_simple vars s
  | Tagged_scrutinee => Some (Value_const (Tagged_immediate i))
  | Not_scrutinee => Some (Value_const (Tagged_immediate (1 - i)))
  | Affine_argument kind offset slope =>
    Some
      (Value_const
         (const_of_int (affine_immediate_kind_to_standard_int kind)
            (affine_value i offset slope)))
  end.

Definition argument_usable (i : Z) (argument : mergeable_argument) : Prop :=
  match argument with
  | Invariant_argument _ | Tagged_scrutinee => True
  | Not_scrutinee => i = 0 ∨ i = 1
  | Affine_argument _ offset slope =>
    in_range immediate_width offset = true
    ∧ in_range immediate_width slope = true
  end.

Definition argument_fits (scrutinee : simple) (i : Z)
    (argument : mergeable_argument) (s : simple) : Prop :=
  argument_usable i argument
  ∧ ∀ vars : env,
      eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)) ->
      eval_simple vars s = argument_value vars i argument.

Lemma affine_value_identity (i : Z) :
  in_range immediate_width i = true -> affine_value i 0 1 = i.
Proof.
  intros Hrange. unfold affine_value, immediate_add, immediate_mul.
  rewrite Z.mul_1_r, Z.add_0_r.
  rewrite (wrap_of_in_range _ i immediate_width_positive Hrange).
  exact (wrap_of_in_range _ i immediate_width_positive Hrange).
Qed.

Lemma recognize_invariant_argument_same (arg : list (Z * simple))
    (s0 : simple) (i : Z) (s : simple) :
  recognize_invariant_argument arg = Some s0 -> (i, s) ∈ arg -> s = s0.
Proof.
  unfold recognize_invariant_argument.
  destruct arg as [ | [i0 s1] rest]; [ discriminate | ].
  destruct (forallb (fun entry => same_simple s1 (snd entry)) rest) eqn:Hall;
    [ | discriminate ].
  intros [= <-] [[= _ ->] | Hin]; [ reflexivity | ].
  rewrite forallb_forall in Hall. specialize (Hall _ Hin). cbn in Hall.
  unfold same_simple in Hall.
  destruct (simple_eq_dec s1 s) as [Heq | ]; [ | discriminate ].
  symmetry. exact Heq.
Qed.

Lemma recognize_mergeable_argument_sound (scrutinee : simple)
    (arg : list (Z * simple)) (argument : mergeable_argument) (i : Z)
    (s : simple) :
  recognize_mergeable_argument scrutinee arg = Some argument ->
  (i, s) ∈ arg ->
  in_range immediate_width i = true ->
  argument_fits scrutinee i argument s.
Proof.
  unfold recognize_mergeable_argument.
  destruct (recognize_invariant_argument arg) as [s0 | ] eqn:Hinv.
  - intros [= <-] Hin _. split; [ exact I | ]. intros vars _. cbn.
    rewrite (recognize_invariant_argument_same arg s0 i s Hinv Hin).
    reflexivity.
  - destruct (fold_left check_arm Maybe_affine arg)
      as [ | | | kind offset slope] eqn:Hfold; try discriminate.
    pose proof (fold_check_arm_seen arg Maybe_affine [] eq_refl) as Hseen.
    unfold fold_left in Hfold. rewrite Hfold in Hseen. cbn [app] in Hseen.
    destruct Hseen as (Hoffset & Hslope & Hfits).
    intros Hargument Hin Hrange. specialize (Hfits _ Hin).
    unfold affine_fits in Hfits. cbn [fst snd] in Hfits.
    destruct ((offset =? 0) && (slope =? 1)) eqn:Hidentity.
    + apply andb_true_iff in Hidentity as [H0 H1].
      apply Z.eqb_eq in H0, H1. subst offset slope.
      rewrite (affine_value_identity i Hrange) in Hfits.
      destruct kind; injection Hargument as <-; (split; [ exact I | ]);
        intros vars Hscrutinee; rewrite Hfits; cbn.
      * reflexivity.
      * symmetry. exact Hscrutinee.
    + destruct kind.
      * match type of Hargument with
        | (if ?test then _ else _) = _ => destruct test eqn:Hnot
        end.
        -- injection Hargument as <-.
           apply andb_true_iff in Hnot as [Hnot Hbools].
           apply andb_true_iff in Hnot as [H1 Hm1].
           apply Z.eqb_eq in H1, Hm1. subst offset slope.
           rewrite forallb_forall in Hbools. specialize (Hbools _ Hin).
           cbn [fst] in Hbools. apply orb_true_iff in Hbools.
           assert (Hi : i = 0 ∨ i = 1)
             by (destruct Hbools as [H | H]; apply Z.eqb_eq in H; auto).
           split; [ exact Hi | ]. intros vars _. rewrite Hfits.
           destruct Hi as [-> | ->]; reflexivity.
        -- injection Hargument as <-. split; [ split; assumption | ].
           intros vars _. rewrite Hfits. reflexivity.
      * injection Hargument as <-. split; [ split; assumption | ].
        intros vars _. rewrite Hfits. reflexivity.
Qed.

Lemma first_arguments_in (rows : list (Z * list simple)) :
  ∀ (arg : list (Z * simple)) (i : Z) (args : list simple),
    first_arguments rows = Some arg ->
    (i, args) ∈ rows ->
    ∃ (s : simple) (rest : list simple),
      args = s :: rest ∧ (i, s) ∈ arg ∧ (i, rest) ∈ other_arguments rows.
Proof.
  induction rows as [ | [i0 args0] rows IH]; intros arg i args Hfirst Hin;
    [ destruct Hin | ].
  simpl in Hfirst. destruct args0 as [ | s0 rest0]; [ discriminate | ].
  destruct (first_arguments rows) as [arg' | ] eqn:Hrows; [ | discriminate ].
  cbn in Hfirst. injection Hfirst as <-.
  destruct Hin as [[= <- <-] | Hin].
  - exists s0, rest0. split; [ reflexivity | ]. split; left; reflexivity.
  - destruct (IH arg' i args eq_refl Hin) as (s & rest & -> & Hs & Hrest).
    exists s, rest. split; [ reflexivity | ]. split; right; assumption.
Qed.

Lemma recognize_mergeable_arguments_sound (scrutinee : simple)
    (num_arms : nat) :
  ∀ (num_args : nat) (rows : list (Z * list simple))
    (arguments : list mergeable_argument) (i : Z) (args : list simple),
    recognize_mergeable_arguments scrutinee num_arms num_args rows
    = Some arguments ->
    (i, args) ∈ rows ->
    length args = num_args ->
    in_range immediate_width i = true ->
    Forall2 (argument_fits scrutinee i) arguments args.
Proof.
  induction num_args as [ | num_args IH];
    intros rows arguments i args Hrec Hin Hlen Hrange.
  - injection Hrec as <-. destruct args; [ constructor | discriminate ].
  - simpl in Hrec.
    destruct (first_arguments rows) as [arg | ] eqn:Hfirst;
      [ | discriminate ].
    cbn in Hrec.
    destruct (recognize_mergeable_argument scrutinee arg) as [argument | ]
      eqn:Harg; [ | discriminate ].
    cbn in Hrec.
    destruct (enough_arms num_arms argument); [ | discriminate ].
    destruct
      (recognize_mergeable_arguments scrutinee num_arms num_args
         (other_arguments rows))
      as [arguments' | ] eqn:Hrest; [ | discriminate ].
    cbn in Hrec. injection Hrec as <-.
    destruct (first_arguments_in rows arg i args Hfirst Hin)
      as (s & rest & -> & Hs & Hrest_in).
    constructor.
    + exact (recognize_mergeable_argument_sound scrutinee arg argument i s
               Harg Hs Hrange).
    + apply (IH (other_arguments rows) arguments' i rest Hrest Hrest_in);
        [ | exact Hrange ].
      simpl in Hlen. injection Hlen as Hlen. exact Hlen.
Qed.

(* The arguments that are the same in every arm are among those of
   the arms, or the scrutinee. *)
Definition avoids (xs : list variable) (s : simple) : Prop :=
  ∀ x : variable, x ∈ xs -> ~ var_free_in_simple x s.

Definition argument_avoids (xs : list variable)
    (argument : mergeable_argument) : Prop :=
  match argument with
  | Invariant_argument s => avoids xs s
  | Tagged_scrutinee | Not_scrutinee | Affine_argument _ _ _ => True
  end.

Lemma recognize_mergeable_argument_avoids (xs : list variable)
    (scrutinee : simple) (arg : list (Z * simple))
    (argument : mergeable_argument) :
  recognize_mergeable_argument scrutinee arg = Some argument ->
  avoids xs scrutinee ->
  (∀ entry : Z * simple, entry ∈ arg -> avoids xs (snd entry)) ->
  argument_avoids xs argument.
Proof.
  unfold recognize_mergeable_argument. intros Hrec Hscrutinee Harg.
  destruct (recognize_invariant_argument arg) as [s0 | ] eqn:Hinv.
  - injection Hrec as <-. unfold recognize_invariant_argument in Hinv.
    destruct arg as [ | [i0 s1] rest]; [ discriminate | ].
    destruct (forallb (fun entry => same_simple s1 (snd entry)) rest);
      [ | discriminate ].
    injection Hinv as <-. exact (Harg (i0, s1) (or_introl eq_refl)).
  - destruct (fold_left check_arm Maybe_affine arg)
      as [ | | | kind offset slope]; try discriminate.
    destruct ((offset =? 0) && (slope =? 1)).
    + destruct kind; injection Hrec as <-; [ exact I | exact Hscrutinee ].
    + destruct kind;
        [ match type of Hrec with
          | (if ?test then _ else _) = _ => destruct test
          end | ];
        injection Hrec as <-; exact I.
Qed.

Lemma first_arguments_from (rows : list (Z * list simple)) :
  ∀ (arg : list (Z * simple)) (entry : Z * simple),
    first_arguments rows = Some arg ->
    entry ∈ arg ->
    ∃ (row : Z * list simple) (rest : list simple),
      row ∈ rows ∧ snd row = snd entry :: rest.
Proof.
  induction rows as [ | [i0 args0] rows IH]; intros arg entry Hfirst Hin.
  - injection Hfirst as <-. destruct Hin.
  - simpl in Hfirst. destruct args0 as [ | s0 rest0]; [ discriminate | ].
    destruct (first_arguments rows) as [arg' | ] eqn:Hrows;
      [ | discriminate ].
    cbn in Hfirst. injection Hfirst as <-.
    destruct Hin as [<- | Hin].
    + exists (i0, s0 :: rest0), rest0. split; [ left | ]; reflexivity.
    + destruct (IH arg' entry eq_refl Hin) as (row & rest & Hrow & Hsnd).
      exists row, rest. split; [ right; exact Hrow | exact Hsnd ].
Qed.

Lemma recognize_mergeable_arguments_avoid (xs : list variable)
    (scrutinee : simple) (num_arms : nat) :
  avoids xs scrutinee ->
  ∀ (num_args : nat) (rows : list (Z * list simple))
    (arguments : list mergeable_argument),
    recognize_mergeable_arguments scrutinee num_arms num_args rows
    = Some arguments ->
    (∀ row : Z * list simple,
      row ∈ rows ->
      ∀ x : variable, x ∈ xs -> ~ var_free_in_simples x (snd row)) ->
    Forall (argument_avoids xs) arguments.
Proof.
  intros Hscrutinee.
  induction num_args as [ | num_args IH]; intros rows arguments Hrec Hrows.
  - injection Hrec as <-. constructor.
  - simpl in Hrec.
    destruct (first_arguments rows) as [arg | ] eqn:Hfirst;
      [ | discriminate ].
    cbn in Hrec.
    destruct (recognize_mergeable_argument scrutinee arg) as [argument | ]
      eqn:Harg; [ | discriminate ].
    cbn in Hrec.
    destruct (enough_arms num_arms argument); [ | discriminate ].
    destruct
      (recognize_mergeable_arguments scrutinee num_arms num_args
         (other_arguments rows))
      as [arguments' | ] eqn:Hrest; [ | discriminate ].
    cbn in Hrec. injection Hrec as <-.
    constructor.
    + apply (recognize_mergeable_argument_avoids xs scrutinee arg argument
               Harg Hscrutinee).
      intros entry Hin x Hx Hfree.
      destruct (first_arguments_from rows arg entry Hfirst Hin)
        as (row & rest & Hrow & Hsnd).
      apply (Hrows row Hrow x Hx). rewrite Hsnd. apply Exists_cons_hd.
      exact Hfree.
    + apply (IH (other_arguments rows) arguments' Hrest).
      intros row Hrow x Hx Hfree. unfold other_arguments in Hrow.
      apply in_map_iff in Hrow as (row0 & <- & Hrow0).
      apply (Hrows row0 Hrow0 x Hx). cbn [snd] in Hfree.
      destruct (snd row0) as [ | s rest]; [ exact Hfree | ].
      apply Exists_cons_tl. exact Hfree.
Qed.

(* ======================================================================== *)
(* Binding the primitives                                                   *)
(* ======================================================================== *)

Definition avoid (xs : list variable) (args : list simple) : Prop :=
  ∀ x : variable, x ∈ xs -> ~ var_free_in_simples x args.

Lemma avoids_incl (xs xs' : list variable) (s : simple) :
  incl xs' xs -> avoids xs s -> avoids xs' s.
Proof. intros Hincl Havoids x Hx. apply Havoids, Hincl, Hx. Qed.

Section Merging.

Variables (statics : static_env) (hp : heap) (rd : read_record)
  (fresh0 : list variable)
  (vars0 : env).

(* What holds as the arguments are rebuilt: the [Let]s made so far run
   from the state of the [Switch] to [vars]; what is known holds
   there; a [Simple] without the names has the value that it had; and
   the names that are left are distinct, among those given, and not in
   what is known. *)
Definition merging_inv (vars : env) (m : merging) : Prop :=
  (∀ (body : expr) (conts : cont_env),
    equivalent
      (State (bind_lets (merging_lets m) body) vars0 conts statics hp rd)
      (State body vars conts statics hp rd))
  ∧ all_computed (merging_known m) vars statics hp rd
  ∧ (∀ s : simple, avoids fresh0 s -> eval_simple vars s = eval_simple vars0 s)
  ∧ NoDup (merging_fresh m)
  ∧ incl (merging_fresh m) fresh0
  ∧ (∀ entry : variable * (prim * list simple),
      entry ∈ merging_known m ->
      ∀ x : variable,
        x ∈ merging_fresh m ->
        fst entry <> x ∧ ~ var_free_in_simples x (snd (snd entry))).

(* What a step of the rebuilding gives: a [Simple] that has a value in
   a state that keeps the invariant, and that is without the names
   that are left. *)
Definition merging_result (vars : env) (m : merging) (s : simple) (v : value)
    (m' : merging) : Prop :=
  ∃ vars' : env,
    merging_inv vars' m'
    ∧ eval_simple vars' s = Some v
    ∧ avoids (merging_fresh m') s
    ∧ (∀ s0 : simple,
        avoids (merging_fresh m) s0 ->
        eval_simple vars' s0 = eval_simple vars s0)
    ∧ incl (merging_fresh m') (merging_fresh m).

Lemma merging_result_same (vars : env) (m : merging) (s : simple)
    (v : value) :
  merging_inv vars m -> eval_simple vars s = Some v ->
  avoids (merging_fresh m) s -> merging_result vars m s v m.
Proof.
  intros Hinv Hs Havoids. exists vars.
  split; [ exact Hinv | ]. split; [ exact Hs | ]. split; [ exact Havoids | ].
  split; [ reflexivity | apply incl_refl ].
Qed.

Lemma merging_result_after (vars vars1 : env) (m m1 : merging) (s : simple)
    (v : value) (m2 : merging) :
  (∀ s0 : simple,
    avoids (merging_fresh m) s0 ->
    eval_simple vars1 s0 = eval_simple vars s0) ->
  incl (merging_fresh m1) (merging_fresh m) ->
  merging_result vars1 m1 s v m2 -> merging_result vars m s v m2.
Proof.
  intros Hextends Hincl (vars2 & Hinv & Hs & Havoids & Hextends' & Hincl').
  exists vars2. split; [ exact Hinv | ]. split; [ exact Hs | ].
  split; [ exact Havoids | ]. split.
  - intros s0 Hs0. rewrite Hextends'; [ apply Hextends, Hs0 | ].
    exact (avoids_incl _ _ s0 Hincl Hs0).
  - intros x Hx. apply Hincl, Hincl', Hx.
Qed.

Lemma bind_prim_sound (vars : env) (m : merging) (kind : flambda_kind)
    (p : prim) (args : list simple) (s : simple) (m' : merging)
    (vs : list value) (v : value) :
  merging_inv vars m ->
  bind_prim m kind p args = Some (s, m') ->
  avoid (merging_fresh m) args ->
  prim_reads_heap p = false ->
  eval_simples vars args = Some vs ->
  (∀ l : location, Value_location l ∉ vs) ->
  eval_prim statics p vs = Some v ->
  kind_of_value v = kind ->
  merging_result vars m s v m'.
Proof using Type.
  intros Hinv Hbind Hargs Hreads Hvs Hnone Hprim Hkind.
  pose proof Hinv as (Hlets & Hknown & Hextends & Hnodup & Hincl & Havoid).
  assert (Hstable : stable_prim p = true).
  { unfold stable_prim. rewrite Hreads. reflexivity. }
  assert (Hnot : ∀ q : heap_primitive, p <> Heap q).
  { intros q ->. discriminate Hreads. }
  (* The primitive runs to its value, and leaves the heap and the
     record, in every environment that gives its arguments these
     values. *)
  assert
    (Hrun :
      ∀ vars' : env,
        eval_simples vars' args = Some vs ->
        exec_named statics hp rd vars' (Prim p args) = Some (v, hp, rd)).
  { intros vars' Hvs'.
    apply (eval_named_exec_named statics hp rd vars' p args vs v Hreads Hvs').
    - intros l Hl. destruct (Hnone l Hl).
    - rewrite eval_named_prim, Hvs'. exact Hprim. }
  unfold bind_prim in Hbind.
  destruct
    (if existsb simple_is_var args
     then computed_by (merging_known m) p args
     else None)
    as [x | ] eqn:Hfound.
  - injection Hbind as <- <-.
    assert (Hin : (x, (p, args)) ∈ merging_known m).
    { destruct (existsb simple_is_var args); [ | discriminate ].
      exact (computed_by_sound _ _ _ _ Hfound). }
    destruct (Hknown _ Hin) as (_ & v' & Hrun' & Hlookup).
    cbn [fst snd] in Hrun', Hlookup.
    rewrite (Hrun vars Hvs) in Hrun'. injection Hrun' as <-.
    apply merging_result_same; [ exact Hinv | exact Hlookup | ].
    intros y Hy Hfree. simpl in Hfree. subst y.
    exact (proj1 (Havoid _ Hin x Hy) eq_refl).
  - destruct (merging_fresh m) as [ | x fresh] eqn:Hfresh; [ discriminate | ].
    injection Hbind as <- <-. subst kind.
    assert (Hx : x ∈ fresh0) by (apply Hincl; left; reflexivity).
    apply NoDup_cons_iff in Hnodup as [Hnotin Hnodup].
    assert (Hargs_x : ~ var_free_in_simples x args)
      by (apply Hargs; left; reflexivity).
    assert
      (Hold :
        all_computed (merging_known m) ((x, v) :: vars) statics hp rd).
    { intros entry Hin.
      destruct (Hknown entry Hin) as (Hstable' & v' & Hrun' & Hlookup).
      destruct (Havoid entry Hin x (or_introl eq_refl)) as [Hne Hfree].
      split; [ exact Hstable' | ]. exists v'. split.
      - rewrite <- Hrun'. apply exec_named_args.
        rewrite eval_simples_extend; [ reflexivity | exact Hfree ].
      - simpl.
        destruct (Pos.eqb_spec (fst entry) x) as [Heq | _];
          [ contradiction | exact Hlookup ]. }
    exists ((x, v) :: vars). cbn [merging_lets merging_known merging_fresh].
    split; [ | split; [ | split; [ | split ] ] ].
    + split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
      * intros body conts. cbn [bind_lets].
        apply
          (equivalent_trans _
             (State (Let x (kind_of_value v) (Prim p args) body) vars conts
                statics hp rd));
          [ apply Hlets | ].
        apply step_equivalent.
        rewrite
          (step_let_no_location _ _ _ _ _ _ _ _ _ rd_ vs Hnot Hvs Hnone).
        rewrite Hprim, kind_eqb_refl. reflexivity.
      * destruct (existsb simple_is_var args); [ | exact Hold ].
        intros entry [<- | Hin]; [ | exact (Hold entry Hin) ].
        split; [ exact Hstable | ]. exists v. cbn [fst snd].
        split; [ apply Hrun; rewrite eval_simples_extend; assumption | ].
        simpl. rewrite Pos.eqb_refl. reflexivity.
      * intros s0 Hs0.
        rewrite eval_simple_extend; [ apply Hextends, Hs0 | apply Hs0, Hx ].
      * exact Hnodup.
      * intros y Hy. apply Hincl. right. exact Hy.
      * intros entry Hin y Hy.
        assert (Hentry : entry = (x, (p, args)) ∨ entry ∈ merging_known m).
        { destruct (existsb simple_is_var args);
            [ destruct Hin as [<- | Hin]; auto | auto ]. }
        destruct Hentry as [-> | Hin'].
        -- cbn [fst snd].
           split; [ intros ->; contradiction | apply Hargs; right; exact Hy ].
        -- apply (Havoid entry Hin'). right. exact Hy.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
    + intros y Hy Hfree. simpl in Hfree. subst y. contradiction.
    + intros s0 Hs0. apply eval_simple_extend. apply Hs0. rewrite Hfresh.
      left. reflexivity.
    + intros y Hy. rewrite Hfresh. right. exact Hy.
Qed.

(* Arithmetic on two immediates of one kind. *)
Lemma eval_immediate_arith (kind : affine_immediate_kind)
    (op : binary_int_arith_op) (a b r : Z) :
  in_range immediate_width a = true ->
  in_range immediate_width b = true ->
  eval_int_arith immediate_width op a b = Some r ->
  eval_prim statics
    (Binary (Int_arith (affine_immediate_kind_to_standard_int kind) op))
    [Value_const
       (const_of_int (affine_immediate_kind_to_standard_int kind) a);
     Value_const
       (const_of_int (affine_immediate_kind_to_standard_int kind) b)]
  = Some
      (Value_const
         (const_of_int (affine_immediate_kind_to_standard_int kind) r)).
Proof.
  intros Ha Hb Hr.
  destruct kind;
    cbn [eval_prim eval_binary_primitive int_of_value
         affine_immediate_kind_to_standard_int const_of_int];
    unfold immediate_width in Ha, Hb, Hr; cbn [int_width] in *;
    rewrite Ha, Hb; cbn; rewrite Hr; reflexivity.
Qed.

Lemma bind_arith_sound (vars : env) (m : merging)
    (kind : affine_immediate_kind) (op : binary_int_arith_op)
    (sa sb : simple) (a b r : Z) (s : simple) (m' : merging) :
  merging_inv vars m ->
  bind_prim m
    (kind_of_standard_int (affine_immediate_kind_to_standard_int kind))
    (Binary (Int_arith (affine_immediate_kind_to_standard_int kind) op))
    [sa; sb]
  = Some (s, m') ->
  avoids (merging_fresh m) sa ->
  avoids (merging_fresh m) sb ->
  eval_simple vars sa
  = Some
      (Value_const
         (const_of_int (affine_immediate_kind_to_standard_int kind) a)) ->
  eval_simple vars sb
  = Some
      (Value_const
         (const_of_int (affine_immediate_kind_to_standard_int kind) b)) ->
  in_range immediate_width a = true ->
  in_range immediate_width b = true ->
  eval_int_arith immediate_width op a b = Some r ->
  merging_result vars m s
    (Value_const
       (const_of_int (affine_immediate_kind_to_standard_int kind) r))
    m'.
Proof.
  intros Hinv Hbind Hsa Hsb Ha Hb Hra Hrb Hr.
  eapply (bind_prim_sound vars m _ _ _ s m' _ _ Hinv Hbind).
  - intros x Hx Hfree. unfold var_free_in_simples in Hfree.
    apply Exists_cons in Hfree as [Hfree | Hfree];
      [ exact (Hsa x Hx Hfree) | ].
    apply Exists_cons in Hfree as [Hfree | Hfree];
      [ exact (Hsb x Hx Hfree) | ].
    apply Exists_nil in Hfree. exact Hfree.
  - reflexivity.
  - cbn [eval_simples]. rewrite Ha, Hb. reflexivity.
  - intros l [Hl | [Hl | [ ]]]; discriminate Hl.
  - exact (eval_immediate_arith kind op a b r Hra Hrb Hr).
  - destruct kind; reflexivity.
Qed.

(* The addition of the offset. *)
Lemma rebuild_affine_final (vars : env) (m : merging)
    (kind : affine_immediate_kind) (offset : Z) (must_negate : bool)
    (scaled : simple) (scaled_v : Z) (s : simple) (m' : merging) :
  merging_inv vars m ->
  (if (offset =? 0) && negb must_negate
   then Some (scaled, m)
   else
     bind_prim m
       (kind_of_standard_int (affine_immediate_kind_to_standard_int kind))
       (Binary
          (Int_arith (affine_immediate_kind_to_standard_int kind)
             (if must_negate then Sub else Add)))
       [Const
          (const_of_int (affine_immediate_kind_to_standard_int kind) offset);
        scaled])
  = Some (s, m') ->
  avoids (merging_fresh m) scaled ->
  eval_simple vars scaled
  = Some
      (Value_const
         (const_of_int (affine_immediate_kind_to_standard_int kind)
            scaled_v)) ->
  in_range immediate_width scaled_v = true ->
  in_range immediate_width offset = true ->
  ∃ r : Z,
    in_range immediate_width r = true
    ∧ congruent r
        (if must_negate then offset - scaled_v else offset + scaled_v)
    ∧ merging_result vars m s
        (Value_const
           (const_of_int (affine_immediate_kind_to_standard_int kind) r))
        m'.
Proof.
  intros Hinv Hfinal Havoids Hscaled Hrange Hoffset.
  destruct ((offset =? 0) && negb must_negate) eqn:Hskip.
  - injection Hfinal as <- <-. apply andb_true_iff in Hskip as [H0 Hnegate].
    apply Z.eqb_eq in H0. apply negb_true_iff in Hnegate. subst.
    exists scaled_v. split; [ exact Hrange | ].
    split; [ replace (0 + scaled_v) with scaled_v by ring; reflexivity | ].
    exact (merging_result_same vars m scaled _ Hinv Hscaled Havoids).
  - exists
      (wrap immediate_width
         (if must_negate then offset - scaled_v else offset + scaled_v)).
    split; [ apply immediate_wrapped | ].
    split; [ apply wrap_congruent | ].
    apply (bind_arith_sound vars m kind _ _ scaled offset scaled_v _ s m' Hinv
             Hfinal).
    + intros x _ Hfree. exact Hfree.
    + exact Havoids.
    + reflexivity.
    + exact Hscaled.
    + exact Hoffset.
    + exact Hrange.
    + destruct must_negate; reflexivity.
Qed.

Lemma rebuild_affine_expr_sound (vars : env) (m : merging)
    (kind : affine_immediate_kind) (offset slope : Z) (scrutinee : simple)
    (i : Z) (s : simple) (m' : merging) :
  merging_inv vars m ->
  rebuild_affine_expr kind offset slope scrutinee m = Some (s, m') ->
  avoids (merging_fresh m) scrutinee ->
  eval_simple vars scrutinee
  = Some
      (Value_const
         (const_of_int (affine_immediate_kind_to_standard_int kind) i)) ->
  in_range immediate_width i = true ->
  in_range immediate_width offset = true ->
  in_range immediate_width slope = true ->
  merging_result vars m s
    (Value_const
       (const_of_int (affine_immediate_kind_to_standard_int kind)
          (affine_value i offset slope)))
    m'.
Proof.
  intros Hinv Hrebuild Havoids Hscrutinee Hi Hoffset Hslope.
  unfold rebuild_affine_expr in Hrebuild.
  set (must_negate := slope <? 0) in *.
  set (slope' := if must_negate then immediate_neg slope else slope) in *.
  assert (Hslope' : in_range immediate_width slope' = true).
  { subst slope'. destruct must_negate;
      [ apply immediate_wrapped | exact Hslope ]. }
  assert (Hcongruent' :
    congruent slope' (if must_negate then - slope else slope)).
  { subst slope'.
    destruct must_negate; [ apply wrap_congruent | reflexivity ]. }
  (* The multiplication by the slope. *)
  assert (Hscaled :
    ∃ (scaled : simple) (m1 : merging) (scaled_v : Z) (vars1 : env),
      (if (offset =? 0) && negb must_negate
       then Some (scaled, m1)
       else
         bind_prim m1
           (kind_of_standard_int (affine_immediate_kind_to_standard_int kind))
           (Binary
              (Int_arith (affine_immediate_kind_to_standard_int kind)
                 (if must_negate then Sub else Add)))
           [Const
              (const_of_int (affine_immediate_kind_to_standard_int kind)
                 offset);
            scaled])
      = Some (s, m')
      ∧ merging_inv vars1 m1
      ∧ eval_simple vars1 scaled
        = Some
            (Value_const
               (const_of_int (affine_immediate_kind_to_standard_int kind)
                  scaled_v))
      ∧ avoids (merging_fresh m1) scaled
      ∧ (∀ s0 : simple,
          avoids (merging_fresh m) s0 ->
          eval_simple vars1 s0 = eval_simple vars s0)
      ∧ incl (merging_fresh m1) (merging_fresh m)
      ∧ in_range immediate_width scaled_v = true
      ∧ congruent scaled_v (i * slope')).
  { destruct (slope' =? 1) eqn:Hone.
    - apply Z.eqb_eq in Hone. cbn in Hrebuild.
      exists scrutinee, m, i, vars.
      split; [ exact Hrebuild | ]. split; [ exact Hinv | ].
      split; [ exact Hscrutinee | ]. split; [ exact Havoids | ].
      split; [ reflexivity | ]. split; [ apply incl_refl | ].
      split; [ exact Hi | ]. rewrite Hone.
      replace (i * 1) with i by ring. reflexivity.
    - match type of Hrebuild with
      | context [bind_prim m ?k ?p ?args] =>
        destruct (bind_prim m k p args) as [[scaled m1] | ] eqn:Hmul;
          [ | discriminate ]
      end.
      cbn in Hrebuild.
      destruct
        (bind_arith_sound vars m kind Mul scrutinee _ i slope' _ scaled m1
           Hinv Hmul Havoids ltac:(intros x _ Hfree; exact Hfree) Hscrutinee
           eq_refl Hi
           Hslope' eq_refl)
        as (vars1 & Hinv1 & Hvalue1 & Havoids1 & Hextends1 & Hincl1).
      exists scaled, m1, (wrap immediate_width (i * slope')), vars1.
      split; [ exact Hrebuild | ]. split; [ exact Hinv1 | ].
      split; [ exact Hvalue1 | ]. split; [ exact Havoids1 | ].
      split; [ exact Hextends1 | ]. split; [ exact Hincl1 | ].
      split; [ apply immediate_wrapped | apply wrap_congruent ]. }
  destruct Hscaled
    as (scaled & m1 & scaled_v & vars1 & Hfinal & Hinv1 & Hvalue1 & Havoids1
        & Hextends1 & Hincl1 & Hrange1 & Hcongruent1).
  destruct
    (rebuild_affine_final vars1 m1 kind offset must_negate scaled scaled_v s
       m' Hinv1 Hfinal Havoids1 Hvalue1 Hrange1 Hoffset)
    as (r & Hr & Hcongruent & Hresult).
  apply (merging_result_after vars vars1 m m1 s _ m' Hextends1 Hincl1).
  replace (affine_value i offset slope) with r; [ exact Hresult | ].
  unfold affine_value, immediate_add, immediate_mul.
  apply (congruent_wrap r _ Hr).
  rewrite wrap_congruent, Hcongruent.
  destruct must_negate; rewrite Hcongruent1, Hcongruent'.
  - replace (offset - i * - slope) with (i * slope + offset) by ring.
    reflexivity.
  - replace (offset + i * slope) with (i * slope + offset) by ring.
    reflexivity.
Qed.

Lemma bind_tagged_scrutinee_sound (vars : env) (m : merging)
    (scrutinee : simple) (i : Z) (s : simple) (m' : merging) :
  merging_inv vars m ->
  bind_tagged_scrutinee scrutinee m = Some (s, m') ->
  avoids (merging_fresh m) scrutinee ->
  eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)) ->
  in_range immediate_width i = true ->
  merging_result vars m s (Value_const (Tagged_immediate i)) m'.
Proof.
  intros Hinv Hbind Havoids Hscrutinee Hi.
  eapply (bind_prim_sound vars m _ _ _ s m' _ _ Hinv Hbind).
  - intros x Hx Hfree. unfold var_free_in_simples in Hfree.
    apply Exists_cons in Hfree as [Hfree | Hfree];
      [ exact (Havoids x Hx Hfree) | ].
    apply Exists_nil in Hfree. exact Hfree.
  - reflexivity.
  - cbn [eval_simples]. rewrite Hscrutinee. reflexivity.
  - intros l [Hl | [ ]]. discriminate Hl.
  - cbn [eval_prim eval_unary_primitive int_of_value].
    unfold immediate_width in Hi. rewrite Hi. reflexivity.
  - reflexivity.
Qed.

Lemma rebuild_mergeable_argument_sound (vars : env) (m : merging)
    (scrutinee : simple) (i : Z) (argument : mergeable_argument) (s : simple)
    (m' : merging) (v : value) :
  merging_inv vars m ->
  rebuild_mergeable_argument scrutinee m argument = Some (s, m') ->
  avoids fresh0 scrutinee ->
  eval_simple vars0 scrutinee = Some (Value_const (Naked_immediate i)) ->
  in_range immediate_width i = true ->
  argument_usable i argument ->
  argument_avoids fresh0 argument ->
  argument_value vars0 i argument = Some v ->
  merging_result vars m s v m'.
Proof.
  intros Hinv Hrebuild Hscrutinee0 Hvalue0 Hi Husable Havoids0 Hv.
  pose proof Hinv as (_ & _ & Hextends & _ & Hincl & _).
  assert (Havoids : avoids (merging_fresh m) scrutinee)
    by exact (avoids_incl _ _ scrutinee Hincl Hscrutinee0).
  assert (Hvalue :
    eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)))
    by (rewrite (Hextends scrutinee Hscrutinee0); exact Hvalue0).
  destruct argument as [s0 | | | kind offset slope]; cbn in Hrebuild, Hv.
  - injection Hrebuild as <- <-.
    apply merging_result_same; [ exact Hinv | | ].
    + rewrite (Hextends s0 Havoids0). exact Hv.
    + exact (avoids_incl _ _ s0 Hincl Havoids0).
  - injection Hv as <-.
    exact (bind_tagged_scrutinee_sound vars m scrutinee i s m' Hinv Hrebuild
             Havoids Hvalue Hi).
  - injection Hv as <-.
    destruct (bind_tagged_scrutinee scrutinee m) as [[tagged m1] | ] eqn:Htag;
      [ | discriminate ].
    cbn in Hrebuild.
    destruct
      (bind_tagged_scrutinee_sound vars m scrutinee i tagged m1 Hinv Htag
         Havoids Hvalue Hi)
      as (vars1 & Hinv1 & Hvalue1 & Havoids1 & Hextends1 & Hincl1).
    apply (merging_result_after vars vars1 m m1 s _ m' Hextends1 Hincl1).
    eapply (bind_prim_sound vars1 m1 _ _ _ s m' _ _ Hinv1 Hrebuild).
    + intros x Hx Hfree. unfold var_free_in_simples in Hfree.
      apply Exists_cons in Hfree as [Hfree | Hfree];
        [ exact (Havoids1 x Hx Hfree) | ].
      apply Exists_nil in Hfree. exact Hfree.
    + reflexivity.
    + cbn [eval_simples]. rewrite Hvalue1. reflexivity.
    + intros l [Hl | [ ]]. discriminate Hl.
    + cbn in Husable. destruct Husable as [-> | ->]; reflexivity.
    + reflexivity.
  - injection Hv as <-. cbn in Husable. destruct Husable as [Hoffset Hslope].
    destruct kind.
    + destruct (bind_tagged_scrutinee scrutinee m) as [[tagged m1] | ]
        eqn:Htag; [ | discriminate ].
      cbn in Hrebuild.
      destruct
        (bind_tagged_scrutinee_sound vars m scrutinee i tagged m1 Hinv Htag
           Havoids Hvalue Hi)
        as (vars1 & Hinv1 & Hvalue1 & Havoids1 & Hextends1 & Hincl1).
      apply (merging_result_after vars vars1 m m1 s _ m' Hextends1 Hincl1).
      exact (rebuild_affine_expr_sound vars1 m1 Affine_tagged offset slope
               tagged i s m' Hinv1 Hrebuild Havoids1 Hvalue1 Hi Hoffset
               Hslope).
    + exact (rebuild_affine_expr_sound vars m Affine_naked offset slope
               scrutinee i s m' Hinv Hrebuild Havoids Hvalue Hi Hoffset
               Hslope).
Qed.

Lemma rebuild_mergeable_arguments_sound (scrutinee : simple) (i : Z) :
  avoids fresh0 scrutinee ->
  eval_simple vars0 scrutinee = Some (Value_const (Naked_immediate i)) ->
  in_range immediate_width i = true ->
  ∀ (arguments : list mergeable_argument) (vars : env) (m : merging)
    (args : list simple) (m' : merging) (vs : list value),
    merging_inv vars m ->
    rebuild_mergeable_arguments scrutinee m arguments = Some (args, m') ->
    Forall (argument_usable i) arguments ->
    Forall (argument_avoids fresh0) arguments ->
    Forall2 (fun argument v => argument_value vars0 i argument = Some v)
      arguments vs ->
    ∃ vars' : env,
      merging_inv vars' m'
      ∧ eval_simples vars' args = Some vs
      ∧ (∀ s0 : simple,
          avoids (merging_fresh m) s0 ->
          eval_simple vars' s0 = eval_simple vars s0)
      ∧ incl (merging_fresh m') (merging_fresh m).
Proof.
  intros Hscrutinee Hvalue Hi.
  induction arguments as [ | argument arguments IH];
    intros vars m args m' vs Hinv Hrebuild Husable Havoid Hvalues.
  - injection Hrebuild as <- <-. inversion Hvalues. subst. exists vars.
    split; [ exact Hinv | ]. split; [ reflexivity | ].
    split; [ reflexivity | apply incl_refl ].
  - simpl in Hrebuild.
    destruct (rebuild_mergeable_argument scrutinee m argument)
      as [[s m1] | ] eqn:Hfirst; [ | discriminate ].
    cbn in Hrebuild.
    destruct (rebuild_mergeable_arguments scrutinee m1 arguments)
      as [[args1 m2] | ] eqn:Hrest; [ | discriminate ].
    cbn in Hrebuild. injection Hrebuild as <- <-.
    inversion Hvalues as [ | argument' v arguments' vs' Hv Hvs]. subst.
    inversion Husable as [ | argument' arguments' Husable1 Husable']. subst.
    inversion Havoid as [ | argument' arguments' Havoid1 Havoid']. subst.
    destruct
      (rebuild_mergeable_argument_sound vars m scrutinee i argument s m1 v
         Hinv Hfirst Hscrutinee Hvalue Hi Husable1 Havoid1 Hv)
      as (vars1 & Hinv1 & Hs & Havoids1 & Hextends1 & Hincl1).
    destruct (IH vars1 m1 args1 m2 vs' Hinv1 Hrest Husable' Havoid' Hvs)
      as (vars2 & Hinv2 & Hargs & Hextends2 & Hincl2).
    exists vars2. split; [ exact Hinv2 | ]. split; [ | split ].
    + cbn [eval_simples]. rewrite (Hextends2 s Havoids1), Hs, Hargs.
      reflexivity.
    + intros s0 Hs0. rewrite Hextends2; [ apply Hextends1, Hs0 | ].
      exact (avoids_incl _ _ s0 Hincl1 Hs0).
    + intros x Hx. apply Hincl1, Hincl2, Hx.
Qed.

End Merging.

(* ======================================================================== *)
(* Of the machine                                                           *)
(* ======================================================================== *)

Lemma name_new_spec (known : computed) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (x : variable) :
  name_new known scrutinee arms x = true ->
  ~ var_free_in_simple x scrutinee
  ∧ (∀ arm : Z * apply_cont,
      arm ∈ arms -> ~ var_free_in_simples x (ac_args (snd arm)))
  ∧ (∀ entry : variable * (prim * list simple),
      entry ∈ known ->
      fst entry <> x ∧ ~ var_free_in_simples x (snd (snd entry))).
Proof.
  unfold name_new. intros Hnew.
  apply andb_true_iff in Hnew as [Hnew Hknown].
  apply andb_true_iff in Hnew as [Hscrutinee Harms].
  apply negb_true_iff in Hscrutinee.
  rewrite forallb_forall in Harms, Hknown.
  split; [ | split ].
  - intros Hfree. apply var_in_simple_sound in Hfree. congruence.
  - intros arm Hin Hfree. specialize (Harms arm Hin).
    apply negb_true_iff in Harms. apply var_in_simples_sound in Hfree.
    congruence.
  - intros entry Hin. specialize (Hknown entry Hin).
    apply andb_true_iff in Hknown as [Hfst Hargs].
    apply negb_true_iff in Hfst. apply negb_true_iff in Hargs.
    apply Pos.eqb_neq in Hfst.
    split; [ intros Heq; apply Hfst; symmetry; exact Heq | ].
    intros Hfree. apply var_in_simples_sound in Hfree.
    exact (eq_true_false_abs _ Hfree Hargs).
Qed.

Lemma mergeable_arms_spec (arms : list (Z * apply_cont))
    (cont : continuation) (num_args : nat) :
  mergeable_arms arms = Some (cont, num_args) ->
  ∀ arm : Z * apply_cont,
    arm ∈ arms ->
    ac_continuation (snd arm) = cont
    ∧ length (ac_args (snd arm)) = num_args.
Proof.
  unfold mergeable_arms. destruct arms as [ | [i0 ac0] rest];
    [ discriminate | ].
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test eqn:Hall
  end; [ | discriminate ].
  intros [= <- <-] arm Hin. rewrite forallb_forall in Hall.
  specialize (Hall arm Hin). apply andb_true_iff in Hall as [Hall Hlength].
  apply andb_true_iff in Hall as [_ Hcont].
  apply Pos.eqb_eq in Hcont. apply Nat.eqb_eq in Hlength.
  split; assumption.
Qed.

(* The values of the arguments of an arm are those of the arguments
   that can be merged. *)
Lemma argument_values (scrutinee : simple) (i : Z) (vars : env) :
  eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)) ->
  ∀ (arguments : list mergeable_argument) (args : list simple)
    (vs : list value),
    Forall2 (argument_fits scrutinee i) arguments args ->
    eval_simples vars args = Some vs ->
    Forall (argument_usable i) arguments
    ∧ Forall2 (fun argument v => argument_value vars i argument = Some v)
        arguments vs.
Proof.
  intros Hscrutinee arguments args vs Hfits. revert vs.
  induction Hfits as [ | argument s arguments args [Husable Hvalue] _ IH];
    intros vs Hvs.
  - injection Hvs as <-. split; constructor.
  - cbn [eval_simples] in Hvs.
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    cbn in Hvs. injection Hvs as <-. destruct (IH vs' eq_refl) as [IH1 IH2].
    split; constructor; try assumption.
    rewrite <- (Hvalue vars Hscrutinee). exact Hs.
Qed.

Theorem merge_arms_refines : merge_arms_refines_statement.
Proof.
  intros known fresh scrutinee arms e vars conts statics hp rd Hmerge Hknown.
  unfold merge_arms in Hmerge.
  destruct (mergeable_arms arms) as [[cont num_args] | ] eqn:Hmergeable;
    [ | discriminate ].
  cbn in Hmerge.
  match type of Hmerge with
  | (if ?test then _ else _) = _ => destruct test eqn:Htests
  end; [ | discriminate ].
  apply andb_true_iff in Htests as [Hranges Hnames].
  unfold names_new in Hnames. apply andb_true_iff in Hnames as [Hdistinct Hnew].
  rewrite forallb_forall in Hranges, Hnew.
  destruct
    (recognize_mergeable_arguments scrutinee (length arms) num_args
       (map (fun arm => (fst arm, ac_args (snd arm))) arms))
    as [arguments | ] eqn:Hrecognize; [ | discriminate ].
  cbn in Hmerge.
  destruct
    (rebuild_mergeable_arguments scrutinee (Merging [] known fresh) arguments)
    as [[args m] | ] eqn:Hrebuild; [ | discriminate ].
  cbn in Hmerge. injection Hmerge as <-.
  destruct (step_switch_cases scrutinee arms vars conts statics hp rd)
    as [Hundefined | (v & i & ac & Hv & Hi & Hfind & Hstep)].
  { apply step_undefined_refines, Hundefined. }
  apply (refines_trans _ (State (Apply_cont ac) vars conts statics hp rd));
    [ apply equivalent_refines, step_equivalent, Hstep | ].
  assert (Hvalue : v = Value_const (Naked_immediate i)).
  { destruct v; try discriminate Hi.
    match goal with
    | c : const |- _ => destruct c; try discriminate Hi
    end.
    injection Hi as ->. reflexivity. }
  subst v. clear Hi.
  pose proof (find_arm_in i arms ac Hfind) as Hin.
  destruct (mergeable_arms_spec arms cont num_args Hmergeable _ Hin)
    as [Hcont Hlength].
  cbn [snd] in Hcont, Hlength.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs.
  2: { apply step_undefined_refines. rewrite step_apply_cont, Hvs.
       reflexivity. }
  assert (Hscrutinee : avoids fresh scrutinee).
  { intros x Hx. exact (proj1 (name_new_spec _ _ _ x (Hnew x Hx))). }
  pose proof
    (recognize_mergeable_arguments_sound scrutinee (length arms) num_args _
       arguments i (ac_args ac) Hrecognize
       (in_map (fun arm => (fst arm, ac_args (snd arm))) arms (i, ac) Hin)
       Hlength (Hranges (i, ac) Hin))
    as Hfits.
  destruct (argument_values scrutinee i vars Hv arguments (ac_args ac) vs Hfits
              Hvs)
    as [Husable Hvalues].
  assert (Havoid : Forall (argument_avoids fresh) arguments).
  { apply (recognize_mergeable_arguments_avoid fresh scrutinee (length arms)
             Hscrutinee num_args _ arguments Hrecognize).
    intros row Hrow x Hx. apply in_map_iff in Hrow as (arm & <- & Harm).
    exact (proj1 (proj2 (name_new_spec _ _ _ x (Hnew x Hx))) arm Harm). }
  assert
    (Hinv : merging_inv statics hp rd fresh vars vars (Merging [] known fresh)).
  { split; [ intros body conts'; apply equivalent_refl | ].
    split; [ exact Hknown | ]. split; [ reflexivity | ].
    split; [ apply distinct_NoDup, Hdistinct | ].
    split; [ apply incl_refl | ].
    intros entry Hentry x Hx.
    exact (proj2 (proj2 (name_new_spec _ _ _ x (Hnew x Hx))) entry Hentry). }
  destruct
    (rebuild_mergeable_arguments_sound statics hp rd fresh vars scrutinee i
       Hscrutinee Hv (Hranges (i, ac) Hin) arguments vars _ args m vs Hinv
       Hrebuild Husable Havoid Hvalues)
    as (vars' & Hinv' & Hargs & _).
  apply
    (refines_trans _
       (State (Apply_cont (Apply_cont_expr cont args None)) vars' conts
          statics hp rd)).
  - apply equivalent_refines, step_eq_equivalent. rewrite !step_apply_cont.
    cbn [ac_args ac_continuation]. rewrite Hvs, Hargs, Hcont. reflexivity.
  - apply equivalent_refines, equivalent_sym. apply (proj1 Hinv').
Qed.

Theorem rebuild_switch_refines : rebuild_switch_refines_statement.
Proof.
  intros known fresh scrutinee arms vars conts statics hp rd Hknown.
  unfold rebuild_switch. destruct arms as [ | arm arms].
  - apply equivalent_refines, switch_no_arms.
  - destruct (merge_arms known fresh scrutinee (arm :: arms)) as [e | ]
      eqn:Hmerge.
    + exact
      (merge_arms_refines known fresh scrutinee _ e vars conts statics _ rd_
               Hmerge Hknown).
    + apply switch_of_arms_refines.
Qed.

Theorem rebuild_switch_nothing_known_refines :
  rebuild_switch_nothing_known_refines_statement.
Proof.
  intros fresh scrutinee arms vars conts statics hp rd.
  apply rebuild_switch_refines. intros entry [].
Qed.

(* ======================================================================== *)
(* After the first two slices                                               *)
(* ======================================================================== *)

Theorem simplify_switch_merging_refines :
  simplify_switch_merging_refines_statement.
Proof.
  intros meet rewrite known fresh tenv s arms e Hmeet Hkeeps Hrewrite Hfits
    Hmem.
  unfold simplify_switch_merging.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ] eqn:Htype; try discriminate.
  cbn.
  destruct (simplify_arms meet rewrite tenv ty arms) as [arms' | ] eqn:Harms;
    [ | discriminate ].
  intros [= <-].
  assert
    (Hfirst :
      expr_refines_under_valid tenv (Switch s arms) (Switch canonical arms')).
  { apply (expr_refines_under_valid_trans tenv _ (Switch s arms')).
    { exact (simplify_arms_refines meet rewrite tenv s canonical ty arms
               arms' Hmeet Hkeeps Hrewrite Hfits Hmem Htype Harms). }
    apply equivalent_under_valid_refines, equivalent_under_valid.
    apply switch_canonical.
    split; [ exact Hmem | exists ty; exact Htype ]. }
  intros vars conts statics hp rd Hvalid Hsat Hknown.
  apply
    (refines_trans _
       (State (Switch canonical arms') vars conts statics hp rd));
    [ exact (Hfirst vars conts statics hp rd Hvalid Hsat) | ].
  apply rebuild_switch_refines. exact Hknown.
Qed.

Theorem expr_refines_under_valid_where_nothing_known :
  expr_refines_under_valid_where_nothing_known_statement.
Proof.
  intros tenv source target. split.
  - intros Hwhere vars conts statics hp rd Hvalid Hsat.
    apply Hwhere; [ exact Hvalid | exact Hsat | intros entry [ ] ].
  - intros Hunder vars conts statics hp rd Hvalid Hsat _.
    exact (Hunder vars conts statics hp rd Hvalid Hsat).
Qed.

Theorem simplify_switch_merging_of_meet_to_prove_refines :
  simplify_switch_merging_of_meet_to_prove_refines_statement.
Proof.
  intros join fuel rewrite known fresh tenv s arms e Hjoin Hrewrite Hfits
    Hmem.
  apply simplify_switch_merging_refines; try assumption.
  - exact (proj1 (FlambdaMeetClosuresProofs.meet_to_prove_sound _ _ Hjoin)).
  - exact (FlambdaMeetTermsProofs.meet_to_prove_keeps_terms _ _ Hjoin).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma forallb_map_with {A B : Type} (f : B -> bool) (g : A -> B)
    (f' : A -> bool) (l : list A) :
  (∀ a : A, f (g a) = f' a) -> forallb f (map g l) = forallb f' l.
Proof.
  intros Hf. induction l as [ | a l IH]; [ reflexivity | ].
  cbn [map forallb]. rewrite Hf, IH. reflexivity.
Qed.

Lemma var_in_simple_renamed (r : renaming) (x : variable) (s : simple) :
  var_in_simple (rename_variable r x) (rename_simple r s) = var_in_simple x s.
Proof.
  destruct s as [y | sym | c]; simpl; [ apply rename_variable_eqb | | ];
    reflexivity.
Qed.

Lemma same_simple_renamed (r : renaming) (s s' : simple) :
  same_simple (rename_simple r s) (rename_simple r s') = same_simple s s'.
Proof.
  unfold same_simple.
  destruct (simple_eq_dec s s') as [-> | Hdiffer].
  - destruct (simple_eq_dec (rename_simple r s') (rename_simple r s'))
      as [_ | Hne]; [ reflexivity | contradiction Hne; reflexivity ].
  - destruct (simple_eq_dec (rename_simple r s) (rename_simple r s'))
      as [Hsame | _]; [ | reflexivity ].
    contradiction Hdiffer. exact (rename_simple_injective r s s' Hsame).
Qed.

Lemma simple_is_var_renamed (r : renaming) (args : list simple) :
  existsb simple_is_var (map (rename_simple r) args)
  = existsb simple_is_var args.
Proof.
  induction args as [ | s args IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite IH. destruct s; reflexivity.
Qed.

(* The action of a renaming on what the merging is made of. *)
Definition rename_rows (r : renaming) (rows : list (Z * list simple))
    : list (Z * list simple) :=
  map (fun row => (fst row, map (rename_simple r) (snd row))) rows.

Definition rename_argument_of_arms (r : renaming) (arg : list (Z * simple))
    : list (Z * simple) :=
  map (fun entry => (fst entry, rename_simple r (snd entry))) arg.

Definition rename_mergeable_argument (r : renaming)
    (argument : mergeable_argument) : mergeable_argument :=
  match argument with
  | Invariant_argument s => Invariant_argument (rename_simple r s)
  | Tagged_scrutinee | Not_scrutinee | Affine_argument _ _ _ => argument
  end.

Definition rename_merging (r : renaming) (m : merging) : merging :=
  Merging
    (map
       (fun binding : variable * flambda_kind * prim * list simple =>
          let '(x, kind, p, args) := binding in
          (rename_variable r x, kind, p, map (rename_simple r) args))
       (merging_lets m))
    (rename_computed r (merging_known m))
    (map (rename_variable r) (merging_fresh m)).

Definition rename_rebuilt (r : renaming) (rebuilt : simple * merging)
    : simple * merging :=
  (rename_simple r (fst rebuilt), rename_merging r (snd rebuilt)).

Lemma mergeable_arms_renamed (r : renaming) (arms : list (Z * apply_cont)) :
  mergeable_arms (rename_arms r arms)
  = option_map
      (fun found => (rename_continuation r (fst found), snd found))
      (mergeable_arms arms).
Proof.
  unfold mergeable_arms, rename_arms.
  destruct arms as [ | [i0 ac0] rest]; [ reflexivity | ].
  cbn [map fst snd rename_apply_cont ac_continuation ac_args ac_trap_action].
  rewrite length_map.
  match goal with
  | |- (if ?test then _ else _) = option_map _ (if ?test' then _ else _) =>
    replace test with test'; [ destruct test'; reflexivity | ]
  end.
  cbn [forallb fst snd rename_apply_cont ac_continuation ac_args
       ac_trap_action].
  rewrite length_map, rename_continuation_eqb. f_equal.
  - destruct (ac_trap_action ac0); reflexivity.
  - symmetry. apply forallb_map_with. intros arm.
    cbn [fst snd rename_apply_cont ac_continuation ac_args ac_trap_action].
    rewrite length_map, rename_continuation_eqb.
    destruct (ac_trap_action (snd arm)); reflexivity.
Qed.

Lemma name_new_renamed (r : renaming) (known : computed) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (x : variable) :
  name_new (rename_computed r known) (rename_simple r scrutinee)
    (rename_arms r arms) (rename_variable r x)
  = name_new known scrutinee arms x.
Proof.
  unfold name_new, rename_arms, rename_computed.
  rewrite var_in_simple_renamed. f_equal; [ f_equal | ].
  - apply forallb_map_with. intros arm. cbn [fst snd rename_apply_cont ac_args].
    rewrite var_in_simples_renamed. reflexivity.
  - apply forallb_map_with. intros entry. cbn [fst snd].
    rewrite rename_variable_eqb, var_in_simples_renamed. reflexivity.
Qed.

Lemma names_new_renamed (r : renaming) (fresh : list variable)
    (known : computed) (scrutinee : simple) (arms : list (Z * apply_cont)) :
  names_new (map (rename_variable r) fresh) (rename_computed r known)
    (rename_simple r scrutinee) (rename_arms r arms)
  = names_new fresh known scrutinee arms.
Proof.
  unfold names_new. f_equal.
  - apply distinct_map_injective. intros x y. apply rename_variable_eqb.
  - apply forallb_map_with. intros x. apply name_new_renamed.
Qed.

Lemma first_arguments_renamed (r : renaming)
    (rows : list (Z * list simple)) :
  first_arguments (rename_rows r rows)
  = option_map (rename_argument_of_arms r) (first_arguments rows).
Proof.
  induction rows as [ | [i args] rows IH]; [ reflexivity | ].
  destruct args as [ | s args]; [ reflexivity | ].
  change (rename_rows r ((i, s :: args) :: rows))
    with
      ((i, rename_simple r s :: map (rename_simple r) args)
       :: rename_rows r rows).
  cbn [first_arguments]. rewrite IH.
  destruct (first_arguments rows); reflexivity.
Qed.

Lemma other_arguments_renamed (r : renaming)
    (rows : list (Z * list simple)) :
  other_arguments (rename_rows r rows) = rename_rows r (other_arguments rows).
Proof.
  unfold other_arguments, rename_rows. rewrite !map_map. apply map_ext.
  intros [i args]. cbn [fst snd]. destruct args; reflexivity.
Qed.

Lemma recognize_invariant_argument_renamed (r : renaming)
    (arg : list (Z * simple)) :
  recognize_invariant_argument (rename_argument_of_arms r arg)
  = option_map (rename_simple r) (recognize_invariant_argument arg).
Proof.
  destruct arg as [ | [i s] rest]; [ reflexivity | ].
  cbn [rename_argument_of_arms map fst snd recognize_invariant_argument].
  rewrite (forallb_map_with _ _ (fun entry => same_simple s (snd entry)));
    [ destruct (forallb _ rest); reflexivity | ].
  intros entry. cbn [snd]. apply same_simple_renamed.
Qed.

Lemma fold_check_arm_renamed (r : renaming) (arg : list (Z * simple)) :
  ∀ affine : affine_argument_of_mergeable_arm,
    List.fold_left check_arm (rename_argument_of_arms r arg) affine
    = List.fold_left check_arm arg affine.
Proof.
  induction arg as [ | [i s] arg IH]; intros affine; [ reflexivity | ].
  cbn [rename_argument_of_arms map List.fold_left fst snd].
  replace (check_arm affine (i, rename_simple r s))
    with (check_arm affine (i, s))
    by (destruct s; reflexivity).
  apply IH.
Qed.

Lemma recognize_mergeable_argument_renamed (r : renaming)
    (scrutinee : simple) (arg : list (Z * simple)) :
  recognize_mergeable_argument (rename_simple r scrutinee)
    (rename_argument_of_arms r arg)
  = option_map (rename_mergeable_argument r)
      (recognize_mergeable_argument scrutinee arg).
Proof.
  unfold recognize_mergeable_argument.
  rewrite recognize_invariant_argument_renamed.
  destruct (recognize_invariant_argument arg) as [s | ]; [ reflexivity | ].
  cbn [option_map]. unfold fold_left. rewrite fold_check_arm_renamed.
  destruct (List.fold_left check_arm arg Maybe_affine)
    as [ | | | kind offset slope]; try reflexivity.
  unfold rename_argument_of_arms.
  rewrite
    (forallb_map_with _ _ (fun entry => (fst entry =? 0) || (fst entry =? 1)))
    by (intros entry; reflexivity).
  destruct ((offset =? 0) && (slope =? 1)); [ destruct kind; reflexivity | ].
  destruct kind; [ | reflexivity ].
  match goal with
  | |- (if ?test then _ else _) = _ => destruct test; reflexivity
  end.
Qed.

Lemma enough_arms_renamed (r : renaming) (num_arms : nat)
    (argument : mergeable_argument) :
  enough_arms num_arms (rename_mergeable_argument r argument)
  = enough_arms num_arms argument.
Proof. destruct argument; reflexivity. Qed.

Lemma recognize_mergeable_arguments_renamed (r : renaming)
    (scrutinee : simple) (num_arms : nat) :
  ∀ (num_args : nat) (rows : list (Z * list simple)),
    recognize_mergeable_arguments (rename_simple r scrutinee) num_arms
      num_args (rename_rows r rows)
    = option_map (map (rename_mergeable_argument r))
        (recognize_mergeable_arguments scrutinee num_arms num_args rows).
Proof.
  induction num_args as [ | num_args IH]; intros rows; [ reflexivity | ].
  cbn [recognize_mergeable_arguments].
  rewrite first_arguments_renamed.
  destruct (first_arguments rows) as [arg | ]; [ | reflexivity ].
  cbn [option_map]. rewrite recognize_mergeable_argument_renamed.
  destruct (recognize_mergeable_argument scrutinee arg) as [argument | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite enough_arms_renamed.
  destruct (enough_arms num_arms argument); [ | reflexivity ].
  rewrite other_arguments_renamed, IH.
  destruct
    (recognize_mergeable_arguments scrutinee num_arms num_args
       (other_arguments rows));
    reflexivity.
Qed.

Lemma bind_prim_renamed (r : renaming) (m : merging) (kind : flambda_kind)
    (p : prim) (args : list simple) :
  bind_prim (rename_merging r m) kind p (map (rename_simple r) args)
  = option_map (rename_rebuilt r) (bind_prim m kind p args).
Proof.
  unfold bind_prim. rewrite simple_is_var_renamed.
  cbn [rename_merging merging_known merging_fresh merging_lets].
  rewrite computed_by_renamed.
  destruct (existsb simple_is_var args).
  - destruct (computed_by (merging_known m) p args) as [x | ];
      [ reflexivity | ].
    cbn [option_map]. destruct (merging_fresh m); reflexivity.
  - destruct (merging_fresh m); reflexivity.
Qed.

Lemma bind_prim_renamed_with_constant (r : renaming) (m : merging)
    (kind : flambda_kind) (p : prim) (s : simple) (c : const) :
  bind_prim (rename_merging r m) kind p [rename_simple r s; Const c]
  = option_map (rename_rebuilt r) (bind_prim m kind p [s; Const c]).
Proof. exact (bind_prim_renamed r m kind p [s; Const c]). Qed.

Lemma bind_prim_renamed_one (r : renaming) (m : merging)
    (kind : flambda_kind) (p : prim) (s : simple) :
  bind_prim (rename_merging r m) kind p [rename_simple r s]
  = option_map (rename_rebuilt r) (bind_prim m kind p [s]).
Proof. exact (bind_prim_renamed r m kind p [s]). Qed.

Lemma rebuild_affine_expr_renamed (r : renaming)
    (kind : affine_immediate_kind) (offset slope : Z) (scrutinee : simple)
    (m : merging) :
  rebuild_affine_expr kind offset slope (rename_simple r scrutinee)
    (rename_merging r m)
  = option_map (rename_rebuilt r)
      (rebuild_affine_expr kind offset slope scrutinee m).
Proof.
  unfold rebuild_affine_expr.
  set (must_negate := slope <? 0).
  set (slope' := if must_negate then immediate_neg slope else slope).
  destruct (slope' =? 1).
  - cbv beta iota.
    destruct ((offset =? 0) && negb must_negate); [ reflexivity | ].
    exact (bind_prim_renamed r m _ _ [Const _; scrutinee]).
  - rewrite bind_prim_renamed_with_constant.
    match goal with
    | |- context [bind_prim m ?k ?p ?args] =>
      destruct (bind_prim m k p args) as [[scaled m1] | ]; [ | reflexivity ]
    end.
    cbn [option_map rename_rebuilt fst snd].
    destruct ((offset =? 0) && negb must_negate); [ reflexivity | ].
    exact (bind_prim_renamed r m1 _ _ [Const _; scaled]).
Qed.

Lemma rebuild_mergeable_argument_renamed (r : renaming) (scrutinee : simple)
    (m : merging) (argument : mergeable_argument) :
  rebuild_mergeable_argument (rename_simple r scrutinee) (rename_merging r m)
    (rename_mergeable_argument r argument)
  = option_map (rename_rebuilt r)
      (rebuild_mergeable_argument scrutinee m argument).
Proof.
  assert (Htag :
    ∀ m : merging,
      bind_tagged_scrutinee (rename_simple r scrutinee) (rename_merging r m)
      = option_map (rename_rebuilt r) (bind_tagged_scrutinee scrutinee m)).
  { intros m0. exact (bind_prim_renamed r m0 _ _ [scrutinee]). }
  destruct argument as [s | | | kind offset slope];
    cbn [rename_mergeable_argument rebuild_mergeable_argument].
  - reflexivity.
  - apply Htag.
  - rewrite Htag.
    destruct (bind_tagged_scrutinee scrutinee m) as [[tagged m1] | ];
      [ | reflexivity ].
    cbn [option_map rename_rebuilt fst snd].
    exact (bind_prim_renamed r m1 _ _ [tagged]).
  - destruct kind; [ | apply rebuild_affine_expr_renamed ].
    rewrite Htag.
    destruct (bind_tagged_scrutinee scrutinee m) as [[tagged m1] | ];
      [ | reflexivity ].
    cbn [option_map rename_rebuilt fst snd].
    apply rebuild_affine_expr_renamed.
Qed.

Lemma rebuild_mergeable_arguments_renamed (r : renaming)
    (scrutinee : simple) (arguments : list mergeable_argument) :
  ∀ m : merging,
    rebuild_mergeable_arguments (rename_simple r scrutinee)
      (rename_merging r m) (map (rename_mergeable_argument r) arguments)
    = option_map
        (fun rebuilt =>
           (map (rename_simple r) (fst rebuilt),
            rename_merging r (snd rebuilt)))
        (rebuild_mergeable_arguments scrutinee m arguments).
Proof.
  induction arguments as [ | argument arguments IH]; intros m;
    [ reflexivity | ].
  cbn [map rebuild_mergeable_arguments].
  rewrite rebuild_mergeable_argument_renamed.
  destruct (rebuild_mergeable_argument scrutinee m argument)
    as [[arg m1] | ]; [ | reflexivity ].
  cbn [option_map rename_rebuilt fst snd]. rewrite IH.
  destruct (rebuild_mergeable_arguments scrutinee m1 arguments)
    as [[args m2] | ]; reflexivity.
Qed.

Lemma bind_lets_renamed (r : renaming)
    (lets : list (variable * flambda_kind * prim * list simple)) :
  ∀ body : expr,
    bind_lets
      (map
         (fun binding : variable * flambda_kind * prim * list simple =>
            let '(x, kind, p, args) := binding in
            (rename_variable r x, kind, p, map (rename_simple r) args))
         lets)
      (rename_expr r body)
    = rename_expr r (bind_lets lets body).
Proof.
  induction lets as [ | [[[x kind] p] args] lets IH]; intros body;
    [ reflexivity | ].
  cbn [map bind_lets]. exact (IH (Let x kind (Prim p args) body)).
Qed.

Theorem merge_arms_rename : merge_arms_rename_statement.
Proof.
  intros r known fresh scrutinee arms. unfold merge_arms.
  rewrite mergeable_arms_renamed.
  destruct (mergeable_arms arms) as [[cont num_args] | ]; [ | reflexivity ].
  cbn [option_map fst snd].
  rewrite names_new_renamed. unfold rename_arms.
  rewrite
    (forallb_map_with _ _
       (fun arm : Z * apply_cont => in_range immediate_width (fst arm)))
    by (intros arm; reflexivity).
  match goal with
  | |- (if ?test then _ else _) = _ => destruct test; [ | reflexivity ]
  end.
  rewrite length_map, map_map.
  replace
    (map
       (fun arm : Z * apply_cont =>
          (fst (fst arm, rename_apply_cont r (snd arm)),
           ac_args (snd (fst arm, rename_apply_cont r (snd arm)))))
       arms)
    with
      (rename_rows r (map (fun arm => (fst arm, ac_args (snd arm))) arms))
    by (unfold rename_rows; rewrite map_map; reflexivity).
  rewrite recognize_mergeable_arguments_renamed.
  destruct
    (recognize_mergeable_arguments scrutinee (length arms) num_args
       (map (fun arm => (fst arm, ac_args (snd arm))) arms))
    as [arguments | ]; [ | reflexivity ].
  cbn [option_map].
  change (Merging [] (rename_computed r known) (map (rename_variable r) fresh))
    with (rename_merging r (Merging [] known fresh)).
  rewrite
    (rebuild_mergeable_arguments_renamed r scrutinee arguments
       (Merging [] known fresh)).
  destruct
    (rebuild_mergeable_arguments scrutinee (Merging [] known fresh) arguments)
    as [[args m] | ]; [ | reflexivity ].
  cbn [option_map fst snd rename_merging merging_lets].
  f_equal.
  exact (bind_lets_renamed r (merging_lets m)
           (Apply_cont (Apply_cont_expr cont args None))).
Qed.

Theorem rebuild_switch_rename : rebuild_switch_rename_statement.
Proof.
  intros r known fresh scrutinee arms. unfold rebuild_switch.
  rewrite merge_arms_rename.
  destruct arms as [ | arm arms]; [ reflexivity | ].
  destruct (merge_arms known fresh scrutinee (arm :: arms)) as [e | ];
    [ reflexivity | ].
  exact (switch_of_arms_rename r scrutinee (arm :: arms)).
Qed.

Theorem simplify_switch_merging_rename :
  simplify_switch_merging_rename_statement.
Proof.
  intros meet rewrite r known fresh tenv s arms Hmeet Hrewrite.
  unfold simplify_switch_merging.
  rewrite type_simple_in_term_exn_equivariant.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ]; try reflexivity.
  cbn [option_map fst snd].
  rewrite (simplify_arms_rename meet rewrite r tenv ty arms Hmeet Hrewrite).
  destruct (simplify_arms meet rewrite tenv ty arms) as [arms' | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite rebuild_switch_rename. reflexivity.
Qed.

Theorem simplify_switch_merging_of_meet_to_prove_rename :
  simplify_switch_merging_of_meet_to_prove_rename_statement.
Proof.
  intros join fuel rewrite r known fresh tenv s arms Hjoin Hrewrite.
  apply simplify_switch_merging_rename; [ | exact Hrewrite ].
  apply meet_to_prove_equivariant, Hjoin.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The arms of the examples apply a continuation that is a variable
   of the statement, and [mergeable_arms] compares it with itself:
   that comparison is rewritten before the rest is computed. *)
Ltac compute_rebuild_switch :=
  unfold rebuild_switch, merge_arms, mergeable_arms;
  cbn [forallb fst snd ac_continuation ac_trap_action ac_args length
       example_index_arms example_tagged_arm];
  rewrite !Pos.eqb_refl; vm_compute; reflexivity.

Theorem example_switch_index : example_switch_index_statement.
Proof.
  intros k.
  assert (Hrebuilt :
    rebuild_switch example_index_known [example_tagged; example_negated]
      (Var example_i) (example_index_arms k)
    = Apply_cont (Apply_cont_expr k [Var example_x] None))
    by compute_rebuild_switch.
  split; [ exact Hrebuilt | ].
  intros vars conts statics hp rd. unfold example_switch_index.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv _ ].
  apply exec_named_pure in Hv as (Hv & -> & ->); [ | reflexivity ].
  rewrite <- Hrebuilt. apply rebuild_switch_refines.
  intros entry [<- | []].
  cbn [eval_named eval_simples eval_simple] in Hv.
  destruct (lookup example_x vars) as [w | ] eqn:Hx; [ | discriminate ].
  cbn [eval_prim eval_unary_primitive] in Hv.
  destruct (int_of_value Standard_tagged_immediate w) as [i | ] eqn:Hint;
    [ | discriminate ].
  cbv beta iota in Hv. injection Hv as <-.
  assert (Hw :
    w = Value_const (Tagged_immediate i)
    ∧ in_range (int_width Standard_naked_immediate) i = true).
  { destruct w; try discriminate Hint.
    match goal with
    | c : const |- _ => destruct c; try discriminate Hint
    end.
    cbn [int_of_value] in Hint.
    match type of Hint with
    | (if ?test then _ else _) = _ => destruct test eqn:Hrange
    end; [ | discriminate ].
    injection Hint as ->. split; [ reflexivity | exact Hrange ]. }
  destruct Hw as [-> Hrange].
  split; [ reflexivity | ].
  exists (Value_const (Tagged_immediate i)). cbn [fst snd].
  split; [ | exact Hx ].
  apply
    (eval_named_exec_named statics hp rd _ _ _
       [Value_const (Naked_immediate i)]).
  - reflexivity.
  - reflexivity.
  - intros l [Hl | [ ]]. discriminate Hl.
  - change
      (eval_prim statics (Unary Tag_immediate)
         [Value_const (Naked_immediate i)]
       = Some (Value_const (Tagged_immediate i))).
    cbn [eval_prim eval_unary_primitive int_of_value]. rewrite Hrange.
    reflexivity.
Qed.

Theorem example_switch_negate : example_switch_negate_statement.
Proof.
  intros k k2.
  apply (let_typed_refines_under_valid _ example_i _ _ _ _ any_naked_bool).
  - vm_compute. reflexivity.
  - reflexivity.
  - intros y [].
  - intros statics hp hp' rd rd' vars v _ Hv. apply any_bool_satisfied.
    apply exec_named_pure in Hv as (Hv & _ & _); [ | reflexivity ].
    cbn [eval_named] in Hv.
    destruct (eval_simples vars _) as [vs | ]; [ | discriminate ].
    exact (int_comp_bool statics _ _ vs v Hv).
  - apply (proj1 (expr_refines_under_valid_where_nothing_known _ _ _)).
    apply (simplify_switch_merging_refines example_meet
             FlambdaSimplifyPassSpec.simplify_apply_cont []
             [example_tagged; example_negated]).
    + exact (proj1 (FlambdaMeetClosuresProofs.meet_to_prove_sound _ _
                      FlambdaMeetExamplesProofs.forgetful_join_sound)).
    + exact (FlambdaMeetTermsProofs.meet_to_prove_keeps_terms _ _
               FlambdaMeetExamplesProofs.forgetful_join_sound).
    + exact simplify_apply_cont_sound.
    + exact example_comparison_env_fits.
    + reflexivity.
    + unfold simplify_switch_merging.
      match goal with
      | |- (let* answer := ?found in _) = _ =>
        let found' := eval vm_compute in found in
        replace found with found' by (vm_compute; reflexivity)
      end.
      cbv beta iota.
      match goal with
      | |- (let* arms := ?simplified in _) = _ =>
        let simplified' := eval vm_compute in simplified in
        replace simplified with simplified' by (vm_compute; reflexivity)
      end.
      cbv beta iota. f_equal. compute_rebuild_switch.
Qed.
